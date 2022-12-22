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
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from rich.progress import track

from idtrackerai import Blob
from idtrackerai.utils import Episode, conf, interpolate_nans


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

    def __init__(
        self, blobs_in_video: list[list[Blob]], bbox_images_path: Path
    ):
        logging.info("Generating ListOfBlobs object")
        self.blobs_in_video = blobs_in_video
        self.bbox_images_path = bbox_images_path
        self.blobs_are_connected = False
        self.number_of_individual_fragments: int

    @property
    def number_of_blobs(self) -> int:
        return sum([len(b_in_frame) for b_in_frame in self.blobs_in_video])

    @property
    def number_of_frames(self):
        return len(self.blobs_in_video)

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
            for (blob_0, blob_1) in itertools.product(
                self.blobs_in_video[frame_i], self.blobs_in_video[frame_i + 1]
            ):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)
        self.blobs_are_connected = True

        # clean cached property
        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                del blob.convexHull

    def save(self, path: Path | str):
        """Saves instance of the class

        Parameters
        ----------
        path_to_save : str, optional
            Path where to save the object, by default None
        """
        logging.info(f"Saving ListOfBlobs at {path}")
        Path(path).parent.mkdir(exist_ok=True)
        if self.blobs_are_connected:
            for blobs_in_frame in self.blobs_in_video:
                for blob in blobs_in_frame:
                    blob.next = []
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

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
        logging.info(f"Loading ListOfBlobs from {path}")
        with open(path, "rb") as file:
            list_of_blobs: ListOfBlobs = pickle.load(file)

        if list_of_blobs.blobs_are_connected:
            logging.info("Reconnecting blobs")
            for blobs_in_frame in list_of_blobs.blobs_in_video:
                for blob in blobs_in_frame:
                    for prev_blob in blob.previous:
                        prev_blob.next.append(blob)
        return list_of_blobs

    # TODO: this should be part of crossing detector.
    # TODO: the term identification_image should be changed.
    def set_images_for_identification(
        self,
        episodes: list[Episode],
        id_images_file_paths: list[Path],
        id_image_size: list[int],
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
        blobs_in_episodes: list[list[Blob]] = Parallel(  # type: ignore
            n_jobs=conf.NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES
        )(
            delayed(self._set_id_images_per_episode)(
                self.bbox_images_path,
                id_image_size[0],
                file,
                episode.index,
                self.blobs_in_video[episode.global_start : episode.global_end],
            )
            for file, episode in track(
                list(zip(id_images_file_paths, episodes)),
                description="Setting images for identification",
            )
        )

        for blobs_in_episode, episode in zip(blobs_in_episodes, episodes):
            self.blobs_in_video[
                episode.global_start : episode.global_end
            ] = blobs_in_episode

    @staticmethod
    def _set_id_images_per_episode(
        bbox_imgs_path: Path,
        id_image_size: int,
        file_path: Path,
        episode_indx: int,
        blobs_in_episode: list[list[Blob]],
    ) -> list[list[Blob]]:
        n_blobs = sum(
            [len(blobs_in_frame) for blobs_in_frame in blobs_in_episode]
        )

        with h5py.File(file_path, "w") as file:
            dataset = file.create_dataset(
                "id_images",
                (n_blobs, id_image_size, id_image_size),
                dtype="uint8",
            )

            index = 0

            for blobs_in_frame in blobs_in_episode:
                for blob in blobs_in_frame:
                    blob.save_image_for_identification(
                        bbox_imgs_path,
                        id_image_size,
                        dataset,
                        index,
                        episode_indx,
                    )
                    index = index + 1
        return blobs_in_episode

    # TODO: maybe move to crossing detector
    def update_id_image_dataset_with_crossings(
        self, id_images_file_paths: list[Path]
    ):
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

        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                id_image_index = blob.id_image_index

                crossings[blob.episode][id_image_index] = blob.is_a_crossing

        for path, crossing in zip(id_images_file_paths, crossings):
            with h5py.File(path, "r+") as file:
                file.create_dataset("crossings", data=crossing)

    # TODO: consider moving to validation
    def next_frame_to_validate(self, current_frame, direction):
        """[Validation] Returns the next frame to be validated.

        Parameters
        ----------
        current_frame : int
            Frame from which to start checking for frames to validate
        direction : string
            Direction towards where to start checking. 'future' will check for
            upcoming frames, and 'past' for previous frames.

        Returns
        -------
        frame_number : int

        """
        logging.debug(f"next_frame_to_validate: {current_frame}")

        if not (
            current_frame > 0 and current_frame < len(self.blobs_in_video)
        ):
            raise Exception(
                "The frame number must be between 0 and the number "
                "of frames in the video"
            )
        if direction == "future":
            blobs_in_frame_to_check = self.blobs_in_video[current_frame + 1 :]
        elif direction == "past":
            blobs_in_frame_to_check = self.blobs_in_video[0:current_frame][
                ::-1
            ]
        for blobs_in_frame in blobs_in_frame_to_check:
            for blob in blobs_in_frame:
                if check_tracking(blobs_in_frame):
                    return blob.frame_number

    # TODO: consider moving to validation
    def interpolate_from_user_generated_centroids(
        self, video, identity, start_frame, end_frame
    ):
        """
        [Validation] Interpolates the centroids of blobs of a given `identity`.

        The interpolation is done using the
        `user_generated_centroids`. The centroid of the blobs without
        user_generated_centroids are assumed to be nan and are interpolated
        accordingly.

        Parameters
        ----------
        video : :class:`video.Video`
            Video object with information of the video to be tracked and the
            tracking process
        identity : int
            Identity of the blobs to be interpolated
        start_frame : int
            Frame from which to start interpolation
        end_frame : int
            Frame where to end the interpolation
        """

        def _check_extreme_blob(extreme_blob):
            if extreme_blob and len(extreme_blob) > 1:
                raise Exception(
                    "The identity must be unique in the first and last frames"
                )
            elif not extreme_blob:
                raise Exception(
                    "There must be a blob with the identity to be\
                                 interpolated in the first and last frames"
                )

        end_frame = end_frame + 1
        if start_frame >= end_frame:
            raise Exception(
                "The initial frame has to be higher than the last frame."
            )

        first_blobs = [
            blob
            for blob in self.blobs_in_video[start_frame]
            if identity in blob.final_identities
        ]
        last_blobs = [
            blob
            for blob in self.blobs_in_video[end_frame - 1]
            if identity in blob.final_identities
        ]

        _check_extreme_blob(first_blobs)
        _check_extreme_blob(last_blobs)

        # Check if they exited or are generated
        both_generated_blobs = (
            first_blobs[0].is_a_generated_blob
            and last_blobs[0].is_a_generated_blob
        )
        both_existed_blobs = (
            not first_blobs[0].is_a_generated_blob
            and not last_blobs[0].is_a_generated_blob
        )

        if not (both_existed_blobs or both_generated_blobs):
            raise Exception(
                "The blobs in the first and last frames should be of the same type, \
                            either generated by the user or by segmentation"
            )

        # Collect centroids of blobs with identity identity that were modified
        # by the user
        centroids_to_interpolate = []
        blobs_of_id = []
        for blobs_in_frame in self.blobs_in_video[start_frame:end_frame]:
            possible_blobs = [
                blob
                for blob in blobs_in_frame
                if identity in blob.final_identities
            ]

            if len(possible_blobs) == 1:
                blobs_of_id.append(possible_blobs[0])
                identity_index = possible_blobs[0].final_identities.index(
                    identity
                )
                fixed_centroid = (None, None)
                if possible_blobs[0].user_generated_centroids is not None:
                    fixed_centroid = possible_blobs[
                        0
                    ].user_generated_centroids[identity_index]
                if fixed_centroid[0] is not None and fixed_centroid[0] > 0:
                    centroids_to_interpolate.append(fixed_centroid)
                elif (
                    possible_blobs[0].user_generated_identities is not None
                    and possible_blobs[0].user_generated_identities[
                        identity_index
                    ]
                    is not None
                ):
                    centroids_to_interpolate.append(
                        possible_blobs[0].final_centroids[identity_index]
                    )
                else:
                    centroids_to_interpolate.append((np.nan, np.nan))
            elif not possible_blobs:
                blobs_of_id.append(None)
                centroids_to_interpolate.append((np.nan, np.nan))
            else:
                raise Exception(
                    "Make sure that the identnties of the user \
                                generated centroids (marked with u-) are unique \
                                in the interpolation interval."
                )

        if not (
            len(centroids_to_interpolate)
            == len(blobs_of_id)
            == end_frame - start_frame
        ):
            raise Exception(
                "The number of user generated centroids before interpolation does \
                            not match the number of frames interpolated."
            )
        if np.isnan(centroids_to_interpolate[0][0]) or np.isnan(
            centroids_to_interpolate[-1][0]
        ):
            raise Exception(
                "The first and last frame of the interpolation interval must contain a \
                            user generated centroid marked with the 'u-' prefix"
            )

        centroids_to_interpolate = np.asarray(centroids_to_interpolate)
        # interpolate linearlry the centroids not generated by the user
        interpolate_nans(centroids_to_interpolate)
        # assign the new centroids to the blobs with identity identity.
        frames = range(start_frame, end_frame)
        for i, (blob, frame) in enumerate(zip(blobs_of_id, frames)):
            if blob is not None:
                identity_index = blob.final_identities.index(identity)
                if blob._user_generated_centroids is None:
                    blob._user_generated_centroids = [(None, None)] * len(
                        blob.final_centroids
                    )
                if blob._user_generated_identities is None:
                    blob._user_generated_identities = [None] * len(
                        blob.final_centroids
                    )
                blob._user_generated_centroids[identity_index] = tuple(
                    centroids_to_interpolate[i, :]
                )
                blob._user_generated_identities[identity_index] = identity
            else:
                if both_existed_blobs:
                    blob_index = np.argmin(
                        [
                            candidate_blob.distance_from_countour_to(
                                tuple(centroids_to_interpolate[i, :])
                            )
                            for candidate_blob in self.blobs_in_video[frame]
                        ]
                    )
                    nearest_blob = self.blobs_in_video[frame][blob_index]
                    nearest_blob.add_centroid(
                        video,
                        tuple(centroids_to_interpolate[i, :]),
                        identity,
                        apply_resolution_reduction=False,
                    )
                elif both_generated_blobs:
                    self.add_blob(
                        video, frame, centroids_to_interpolate[i, :], identity
                    )

        video.is_centroid_updated = True

    # TODO: Consider moving to validation
    def reset_user_generated_identities_and_centroids(
        self, video, start_frame, end_frame, identity=None
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
                "Initial frame number must be smaller than"
                "the final frame number"
            )
        if not (identity is None or identity >= 0):
            # missing identity <= self.number_of_animals but the attribute
            # does not exist
            raise Exception(
                "Identity must be None, zero or a positive integer"
            )

        for blobs_in_frame in self.blobs_in_video[start_frame : end_frame + 1]:
            if identity is None:
                # Reset all user generated identities and centroids
                for blob in blobs_in_frame:
                    if blob.is_a_generated_blob:
                        self.blobs_in_video[blob.frame_number].remove(blob)
                    else:
                        blob._user_generated_identities = None
                        blob._user_generated_centroids = None
            else:
                possible_blobs = [
                    blob
                    for blob in blobs_in_frame
                    if identity in blob.final_identities
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
                            if blob._user_generated_centroids is not None:
                                blob._user_generated_centroids[index] = (
                                    None,
                                    None,
                                )
                            if blob._user_generated_identities is not None:
                                blob._user_generated_identities[index] = None

        video._is_centroid_updated = any(
            [
                any(
                    [
                        cent[0] is not None
                        for cent in blob.user_generated_centroids
                    ]
                )
                for blobs_in_frame in self.blobs_in_video
                for blob in blobs_in_frame
                if blob.user_generated_centroids is not None
            ]
        )

    # TODO: Consider moving to validation
    def add_blob(
        self,
        video,
        frame_number,
        centroid,
        identity,
        apply_resolution_reduction=True,
    ):
        """[Validation] Adds a Blob object the frame number.

        Adds a Blob object to a given frame_number with a given centroid and
        identity. Note that this Blob won't have most of the features (e.g.
        area, contour, fragment_identifier, bounding_box, ...). It is only
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
        apply_resolution_reduction : bool, optional
            Indicates whether resolution reduction must be applied to the given
            centroid, by default True

        Raises
        ------
        Exception
            If the `centroid` is not a tuple of length 2.
        Exception
            If the `identity` is not a number between 1 and the number of
            animals in the video.
        """
        logging.info("Calling add_blob")
        if apply_resolution_reduction:
            centroid = (
                centroid[0] * video.resolution_reduction,
                centroid[1] * video.resolution_reduction,
            )
        if not (isinstance(centroid, tuple) and len(centroid) == 2):
            raise Exception("The centroid must be a tuple of length 2")
        if not (
            isinstance(identity, int)
            and identity > 0
            and identity <= video.number_of_animals
        ):
            raise Exception(
                "The identity must be an integer between 1 and the number of "
                "animals in the video"
            )

        new_blob = Blob(
            centroid=None,
            contour=None,
            area=None,
            bounding_box_in_frame_coordinates=None,
        )
        new_blob._user_generated_centroids = [(centroid[0], centroid[1])]
        new_blob._user_generated_identities = [identity]
        new_blob.frame_number = frame_number
        new_blob._is_an_individual = True
        new_blob._is_a_crossing = False
        new_blob._resolution_reduction = video.resolution_reduction
        new_blob.number_of_animals = video.number_of_animals
        self.blobs_in_video[frame_number].append(new_blob)
        video._is_centroid_updated = True

    @property
    def maximum_number_of_blobs(self):
        return max([len(bl_in_frame) for bl_in_frame in self.blobs_in_video])


# TODO: consider moving to validation
def check_tracking(blobs_in_frame):
    """Returns True if the list of blobs `blobs_in_frame` needs to be
    validated.

    A list of blobs of a frame need to be validated if some blobs are crossings
    or if there is some missing identity.

    Parameters
    ----------
    blobs_in_frame : list
        List of Blob objects in a given frame of the video.

    Returns
    -------
    check_tracking_flag : boolean
    """
    there_are_crossings = any(
        [blob.is_a_crossing for blob in blobs_in_frame]
    )  # check whether there is a crossing in the frame
    missing_identity = any(
        [
            None in blob.final_identities or 0 in blob.final_identities
            for blob in blobs_in_frame
        ]
    )  # Check whether there is some missing identities (0 or None)
    return there_are_crossings or missing_identity
