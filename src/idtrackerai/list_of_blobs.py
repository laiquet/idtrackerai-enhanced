# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)
import itertools
import logging
import pickle
from contextlib import suppress
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from . import Blob
from .utils import Episode, clean_attrs, conf, resolve_path, track


class ListOfBlobs:
    """Contains all the instances of the class :class:`~blob.Blob` for all
    frames in the video.

    Notes
    -----
    Only frames in the tracking interval defined by the user can have blobs.
    The frames ouside of such interval will be empty.


    Parameters
    ----------
    blobs_in_video : list
        List of lists of blobs. Each element in the outer list represents
        a frame. Each elemtn in each inner list represents a blob in
        the frame.
    """

    def __init__(self, blobs_in_video: list[list[Blob]]):
        logging.info("Generating ListOfBlobs object")
        self.blobs_in_video = blobs_in_video
        self.blobs_are_connected = False
        self.number_of_individual_fragments: int

    @property
    def all_blobs(self):
        return itertools.chain.from_iterable(self.blobs_in_video)

    @property
    def number_of_blobs(self) -> int:
        return sum(map(len, self.blobs_in_video))

    @property
    def number_of_crossing_blobs(self) -> int:
        return sum(blob.is_a_crossing for blob in self.all_blobs)

    @property
    def number_of_frames(self):
        return len(self.blobs_in_video)

    @property
    def max_number_of_blobs_in_one_frame(self):
        return max(map(len, self.blobs_in_video))

    def __len__(self):
        return len(self.blobs_in_video)

    def compute_overlapping_between_subsequent_frames(self):
        """Computes overlapping between blobs in consecutive frames.

        Two blobs in consecutive frames overlap if the intersection of the list
        of pixels of both blobs is not empty.

        See Also
        --------
        :meth:`blob.Blob.overlaps_with`
        """

        logging.info("Connecting list of blobs ")

        if self.blobs_are_connected:
            logging.error("List of blobs is already connected")
            return
        # self.disconnect()

        for frame_i in track(range(self.number_of_frames - 1), "Connecting blobs"):
            for blob_0, blob_1 in itertools.product(
                self.blobs_in_video[frame_i], self.blobs_in_video[frame_i + 1]
            ):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)
        self.blobs_are_connected = True

        # clean cached property
        with suppress(AttributeError):
            for blob in self.all_blobs:
                del blob.convexHull

    def save(self, path: Path | str):
        """Saves instance of the class

        Parameters
        ----------
        path_to_save : str, optional
            Path where to save the object, by default None
        """
        path = resolve_path(path)
        logging.info(f"Saving ListOfBlobs at {path}")
        path.parent.mkdir(exist_ok=True)
        self.disconnect()

        for blob in self.all_blobs:
            clean_attrs(blob)

        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.reconnect()

    @classmethod
    def load(cls, path: Path | str) -> "ListOfBlobs":
        """Loads an instance of a class saved in a .npy file.

        Parameters
        ----------
        blob_list_file : Path
            path to a saved instance of a ListOfBlobs object

        Returns
        -------
        ListOfBlobs
        """
        path = resolve_path(path)
        logging.info(f"Loading ListOfBlobs from {path}")
        if not path.is_file():
            v4_path = path.with_name(
                path.name.replace("list_of_blobs", "blobs_collection")
            ).with_suffix(".npy")

            if v4_path.is_file():
                list_of_blobs = cls.load_from_v4(v4_path)
            else:
                raise FileNotFoundError(path)
        else:
            with open(path, "rb") as file:
                list_of_blobs: ListOfBlobs = pickle.load(file)
        list_of_blobs.reconnect()
        return list_of_blobs

    @classmethod
    def load_from_v4(cls, path: Path) -> "ListOfBlobs":
        logging.info("Loading from v4 file: %s", path)
        list_of_blobs: "ListOfBlobs" = np.load(path, allow_pickle=True).item()

        for blob in track(
            list_of_blobs.all_blobs, "Updating objects from an old idtracker.ai version"
        ):
            blob.is_an_individual = blob._is_an_individual  # type:ignore
            blob.fragment_identifier = blob._fragment_identifier  # type:ignore
            blob.blob_index = blob._blob_index  # type:ignore
            blob.identity = blob._identity  # type:ignore
            blob.identity_corrected_solving_jumps = (
                blob._identity_corrected_solving_jumps  # type:ignore
            )
            blob.identities_corrected_closing_gaps = (
                blob._identities_corrected_closing_gaps  # type:ignore
            )
        return list_of_blobs

    def disconnect(self):
        if self.blobs_are_connected:
            for blob in self.all_blobs:
                blob.next.clear()

    def reconnect(self):
        if self.blobs_are_connected:
            for blob in self.all_blobs:
                for prev_blob in blob.previous:
                    prev_blob.next.append(blob)

    # TODO: this should be part of crossing detector.
    # TODO: the term identification_image should be changed.
    def set_images_for_identification(
        self,
        episodes: list[Episode],
        id_images_file_paths: list[Path],
        id_image_size: list[int],
        bbox_images_path: Path,
    ):
        """Computes and saves the images used to classify blobs as crossings
        and individuals and to identify the animals along the video.

        Parameters
        ----------
        episodes_start_end : list
            List of tuples of integers indncating the starting and ending
            frames of each episode.
        id_images_file_paths : list
            List of strings indicating the paths to the files where the
            identification images of each episode are stored.
        id_image_size : tuple
            Tuple indicating the width, height and number of channels of the
            identification images.
        number_of_animals : int
            Number of animals to be tracked as indicated by the user.
        number_of_frames : int
            Number of frames in the video
        video_path : str
            Path to the video file
        height : int
            Height of a video frame considering the resolution reduction
            factor.
        width : int
            Width of a video frame considering the resolution reduction factor.
        """

        inputs = [
            (
                bbox_images_path / f"episode_images_{episode.index}.hdf5",
                id_image_size[0],
                file,
                episode,
                self.blobs_in_video[episode.global_start : episode.global_end],
            )
            for file, episode in zip(id_images_file_paths, episodes)
        ]

        with Pool(conf.number_of_parallel_workers) as p:
            for blobs_in_episode, episode in track(
                p.imap_unordered(self.set_id_images_per_episode, inputs),
                "Setting images for identification",
                len(inputs),
            ):
                self.blobs_in_video[episode.global_start : episode.global_end] = (
                    blobs_in_episode
                )

    @staticmethod
    def set_id_images_per_episode(
        inputs: tuple[Path, int, Path, Episode, list[list[Blob]]]
    ) -> tuple[list[list[Blob]], Episode]:
        bbox_imgs_path, id_image_size, file_path, episode, blobs_in_episode = inputs

        imgs_to_save = np.empty(
            (sum(map(len, blobs_in_episode)), id_image_size, id_image_size), np.uint8
        )

        for index, blob in enumerate(itertools.chain.from_iterable(blobs_in_episode)):
            imgs_to_save[index] = blob.get_image_for_identification(
                id_image_size, bbox_imgs_path
            )
            blob.id_image_index = index
            blob.episode = episode.index

        with h5py.File(file_path, "w") as file:
            file.create_dataset("id_images", data=imgs_to_save)

        return blobs_in_episode, episode

    # TODO: maybe move to crossing detector
    def update_id_image_dataset_with_crossings(self, id_images_file_paths: list[Path]):
        """Adds a array to the identification images files indicating whether
        each image is an individual or a crossing.

        Parameters
        ----------
        video : :class:`idtrackerai.video.Video`
            Video object with information about the video and the tracking
            process.
        """
        logging.info("Updating crossings in identification images files")

        crossings = []
        for path in id_images_file_paths:
            with h5py.File(path, "r") as file:
                crossings.append(np.empty(file["id_images"].shape[0], bool))

        for blob in self.all_blobs:
            id_image_index = blob.id_image_index

            crossings[blob.episode][id_image_index] = blob.is_a_crossing

        for path, crossing in zip(id_images_file_paths, crossings):
            with h5py.File(path, "r+") as file:
                file.create_dataset("crossings", data=crossing)

    def remove_centroid(self, frame_number: int, centroid_to_remove, id_to_remove):
        for blob in self.blobs_in_video[frame_number]:
            for indx, (id, centroid) in enumerate(
                zip(blob.all_final_identities, blob.all_final_centroids)
            ):
                if id == id_to_remove:
                    dist = (centroid[0] - centroid_to_remove[0]) ** 2 + (
                        centroid[1] - centroid_to_remove[1]
                    ) ** 2
                    if dist < 1:  # it is the same centroid
                        blob.init_validator_variables()
                        blob.user_generated_centroids[indx] = (-1, -1)
                        blob.user_generated_identities[indx] = -1

    def reset_user_generated_corrections(
        self, start_frame: int = 0, end_frame: Optional[int] = None
    ):
        """[Validation] Resets the identities and centroids generetad by the user.

        Parameters
        ----------
        start_frame : int
            Frame from which to start resetting identities and centroids
        end_frame : int
            Frame where to end resetting identities and centroids
        """

        for blobs_in_frame in track(
            self.blobs_in_video[start_frame:end_frame], "Resetting user corrections"
        ):
            # Reset all user generated identities and centroids
            for blob in blobs_in_frame:
                if blob.added_by_user:
                    self.blobs_in_video[blob.frame_number].remove(blob)
                else:
                    blob.user_generated_identities = None
                    blob.user_generated_centroids = None

    def update_centroid(
        self, frame_number: int, centroid_id: int, old_centroid, new_centroid
    ):
        old_centroid = tuple(old_centroid)
        new_centroid = tuple(new_centroid)
        blobs_in_frame = self.blobs_in_video[frame_number]
        assert blobs_in_frame

        dist_to_old_centroid: list[tuple[Blob, float]] = []

        for blob in blobs_in_frame:
            try:
                indx, centroid, dist = blob.index_and_centroid_closer_to(
                    old_centroid, centroid_id
                )
            except ValueError:  # blob has not centroid_id
                pass
            else:
                dist_to_old_centroid.append((blob, dist))

        blob_with_old_centroid = min(dist_to_old_centroid, key=lambda x: x[1])[0]
        blob_with_old_centroid.update_centroid(old_centroid, new_centroid, centroid_id)

    def add_centroid(self, frame_number: int, identity: int, centroid):
        centroid = tuple(centroid)
        blobs_in_frame = self.blobs_in_video[frame_number]
        if not blobs_in_frame:
            self.add_blob(frame_number, centroid, identity)
            return

        for blob in blobs_in_frame:
            if blob.contains_point(centroid):
                blob.add_centroid(centroid, identity)
                return

        blob = min(blobs_in_frame, key=lambda b: b.distance_from_countour_to(centroid))
        blob.add_centroid(centroid, identity)

    def add_blob(self, frame_number: int, centroid: tuple, identity: int):
        """[Validation] Adds a Blob object the frame number.

        Adds a Blob object to a given frame_number with a given centroid and
        identity. Note that this Blob won't have most of the features (e.g.
        area, contour, fragment_identifier, bbox, ...). It is only
        intended to be used for validation and correction of trajectories.
        The new blobs generated are considered to be individuals.

        Parameters
        ----------
        frame_number : int
            Frame in which the new blob will be added
        centroid : tuple
            The centroid of the new blob
        identity : int
            Identity of the new blob
        """
        contour = np.array(
            [
                [centroid[0] - 2, centroid[1] - 2],
                [centroid[0] - 2, centroid[1] + 2],
                [centroid[0] + 2, centroid[1] + 2],
                [centroid[0] + 2, centroid[1] - 2],
            ],
            int,
        )
        new_blob = Blob(contour, frame_number)
        new_blob.added_by_user = True
        new_blob.user_generated_centroids = [(centroid[0], centroid[1])]
        new_blob.user_generated_identities = [identity]
        new_blob.is_an_individual = True
        self.blobs_in_video[frame_number].append(new_blob)

    @property
    def maximum_number_of_blobs(self):
        return max(map(len, self.blobs_in_video))
