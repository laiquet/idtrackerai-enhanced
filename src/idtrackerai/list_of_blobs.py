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
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from rich.progress import track

from idtrackerai import Blob
from idtrackerai.utils import Episode, conf, resolve_path


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

        for frame_i in track(
            range(self.number_of_frames - 1), description="Connecting blobs "
        ):
            for blob_0, blob_1 in itertools.product(
                self.blobs_in_video[frame_i], self.blobs_in_video[frame_i + 1]
            ):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)
        self.blobs_are_connected = True

        # clean cached property
        for blob in self.all_blobs:
            try:
                del blob.convexHull
            except AttributeError:
                # Some blob'b bboxes do not overlap with any other blob so their
                # convexHull is not computed
                pass

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
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.reconnect()

    @staticmethod
    def load(path: Path | str) -> "ListOfBlobs":
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
        with open(path, "rb") as file:
            list_of_blobs: ListOfBlobs = pickle.load(file)
        list_of_blobs.reconnect()
        return list_of_blobs

    def get_deep_copy(self) -> "ListOfBlobs":
        self.disconnect()
        copy_of = deepcopy(self)
        self.reconnect()
        copy_of.reconnect()
        return copy_of

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

        with Pool(min(conf.number_of_parallel_workers, len(episodes))) as p:
            for blobs_in_episode, episode in track(
                p.imap_unordered(self._set_id_images_per_episode, inputs),
                "Setting images for identification",
                len(inputs),
            ):
                self.blobs_in_video[
                    episode.global_start : episode.global_end
                ] = blobs_in_episode

    @staticmethod
    def _set_id_images_per_episode(
        inputs: tuple[Path, int, Path, Episode, list[list[Blob]]]
    ) -> tuple[list[list[Blob]], Episode]:
        (bbox_imgs_path, id_image_size, file_path, episode, blobs_in_episode) = inputs

        n_blobs = sum(len(blobs_in_frame) for blobs_in_frame in blobs_in_episode)

        with h5py.File(file_path, "w") as file:
            dataset = file.create_dataset(
                "id_images", (n_blobs, id_image_size, id_image_size), dtype="uint8"
            )

            index = 0

            for blob in itertools.chain.from_iterable(blobs_in_episode):
                blob.save_image_for_identification(
                    bbox_imgs_path, id_image_size, dataset, index, episode.index
                )
                index = index + 1
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

    # TODO: Consider moving to validation
    def reset_user_generated_identities_and_centroids(
        self, start_frame, end_frame, identity=None
    ):
        """
        [Validation] Resets the identities and centroids generetad by the user.

        Resets the identities and centroids generetad by the user to the ones
        computed by the tracking algorithm.

        Parameters
        ----------
        video : :class:`video.Video`
            Video object with information of the video to be tracked and the
            tracking process
        start_frame : int
            Frame from which to start reseting identities and centroids
        end_frame : int
            Frame where to end reseting identities and centroids
        identity : int, optional
            Identity of the blobs to be reseted (default None). If None,
            all the blobs are reseted
        """
        if start_frame > end_frame:
            raise Exception(
                "Initial frame number must be smaller than" "the final frame number"
            )
        if not (identity is None or identity >= 0):
            # missing identity <= self.number_of_animals but the attribute
            # does not exist
            raise Exception("Identity must be None, zero or a positive integer")

        for blobs_in_frame in self.blobs_in_video[start_frame : end_frame + 1]:
            if identity is None:
                # Reset all user generated identities and centroids
                for blob in blobs_in_frame:
                    if blob.is_a_generated_blob:
                        self.blobs_in_video[blob.frame_number].remove(blob)
                    else:
                        blob.user_generated_identities = None
                        blob.user_generated_centroids = None
            else:
                possible_blobs = [
                    blob for blob in blobs_in_frame if identity in blob.final_identities
                ]
                for blob in possible_blobs:
                    if blob.is_a_generated_blob:
                        self.blobs_in_video[blob.frame_number].remove(blob)
                    else:
                        indices = [
                            i
                            for i, final_id in enumerate(blob.final_identities)
                            if final_id == identity
                        ]
                        for index in indices:
                            if blob.user_generated_centroids is not None:
                                blob.user_generated_centroids[index] = (None, None)
                            if blob.user_generated_identities is not None:
                                blob.user_generated_identities[index] = None

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

    def add_centroid(self, frame_number: int, id: int, centroid):
        centroid = tuple(centroid)
        blobs_in_frame = self.blobs_in_video[frame_number]
        if not blobs_in_frame:
            # add blob
            raise NotImplementedError

        for blob in blobs_in_frame:
            if blob.contains_point(centroid):
                blob.add_centroid(centroid, id)
                return

        blob = min(blobs_in_frame, key=lambda b: b.distance_from_countour_to(centroid))
        blob.add_centroid(centroid, id)

    def add_blob(self, frame_number: int, centroid, identity):
        """[Validation] Adds a Blob object the frame number.

        Adds a Blob object to a given frame_number with a given centroid and
        identity. Note that this Blob won't have most of the features (e.g.
        area, contour, fragment_identifier, bbox, ...). It is only
        intended to be used for validation and correction of trajectories.
        The new blobs generated are considered to be individuals.

        Args:
            frame_number (int): frame number where the Blob
            centroid (tuple): tuple with two float number (x, y).
            identity (int): identity of the blob

        Raises:
            Exception: If `identity` is greater of the number of animals in the
            video.

        Parameters
        ----------
        video : :class:`video.Video`
            Video object with information of the video to be tracked and the
            tracking process
        frame_number : int
            Frame in which the new blob will be added
        centroid : tuple
            The centroid of the new blob
        identity : int
            Identity of the new blob

        Raises
        ------
        Exception
            If the `centroid` is not a tuple of length 2.
        Exception
            If the `identity` is not a number between 1 and the number of
            animals in the video.
        """
        contour = np.array(
            [
                [centroid[0] - 1, centroid[1] - 1],
                [centroid[0] - 1, centroid[1] + 1],
                [centroid[0] + 1, centroid[1] + 1],
                [centroid[0] + 1, centroid[1] - 1],
            ]
        )
        new_blob = Blob(contour, frame_number)
        new_blob.user_generated_centroids = [(centroid[0], centroid[1])]
        new_blob.user_generated_identities = [identity]
        new_blob.is_an_individual = True
        self.blobs_in_video[frame_number].append(new_blob)

    @property
    def maximum_number_of_blobs(self):
        return max(map(len, self.blobs_in_video))
