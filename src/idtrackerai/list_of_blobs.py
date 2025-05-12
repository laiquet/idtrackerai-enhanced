import logging
import pickle
from collections.abc import Iterable, Iterator, Sequence
from io import BytesIO
from itertools import chain, pairwise, product
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from . import Blob
from .utils import (
    LOGGING_QUEUE,
    Episode,
    clean_attrs,
    deprecated,
    open_track,
    resolve_path,
    setup_logging_queue,
    track,
)


class _chain(chain):
    """A re-implementation of itertools.chain with added __len__ method for pretty progress bars"""

    len: int | None

    def __len__(self) -> int | None:
        return self.len

    @classmethod
    def from_iterable_of_sequences(cls, iterable: Iterable[Sequence]) -> Iterator:
        chain = cls.from_iterable(iterable)
        chain.len = sum(map(len, iterable))  # type: ignore (not sure how this should be coded)
        return chain


class ListOfBlobs:
    """Contains all the instances of the class :class:`Blob`."""

    blobs_in_video: list[list[Blob]]
    """Main and only attribute of the class, it contains a list of all
    :class:`Blob` instances in a frame for every frame in the video"""

    def __init__(self, blobs_in_video: list[list[Blob]]):
        logging.info("Generating ListOfBlobs object")
        self.blobs_in_video = blobs_in_video

    @classmethod
    def load(cls, file_path: Path | str, verbose: bool = True) -> "ListOfBlobs":
        """Loads an instance of a class saved in a .pickle file.

        Parameters
        ----------
        file_path : Path | str
            Path to a saved instance of a ListOfBlobs object

        Returns
        -------
        ListOfBlobs
        """
        file_path = resolve_path(file_path)
        logging.info(f"Loading ListOfBlobs from {file_path}", stacklevel=2)
        if not file_path.is_file():
            v4_path = file_path.with_name(
                file_path.name.replace("list_of_blobs", "blobs_collection")
            ).with_suffix(".npy")

            if not v4_path.is_file():
                raise FileNotFoundError(file_path)
            else:
                file_path = v4_path

        if file_path.suffix == ".npy":
            list_of_blobs = cls._load_from_v4(file_path, verbose=verbose)
        else:
            with open_track(file_path, "rb", verbose=verbose) as file:
                list_of_blobs: ListOfBlobs = pickle.load(file)
        list_of_blobs.reconnect()
        return list_of_blobs

    def save(self, file_path: Path | str) -> None:
        """Saves instance of the class with Python's pickle protocol

        Parameters
        ----------
        file_path : Path | str
            Path where to save the object
        """
        file_path = resolve_path(file_path)
        logging.info(f"Saving ListOfBlobs at {file_path}", stacklevel=2)
        file_path.parent.mkdir(exist_ok=True)
        self.disconnect()

        for blob in self.all_blobs:
            clean_attrs(blob)

        with open_track(file_path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.reconnect()

    @classmethod
    def _load_from_v4(cls, path: Path, verbose: bool = True) -> "ListOfBlobs":
        logging.info("Loading from v4 file: %s", path)
        with open_track(path, "rb", verbose=verbose) as file:
            list_of_blobs: "ListOfBlobs" = np.load(file, allow_pickle=True).item()

        for blob in track(
            list_of_blobs.all_blobs, "Updating objects from an old idtracker.ai version"
        ):
            blob.is_an_individual = blob._is_an_individual  # type:ignore
            blob.fragment_identifier = blob._fragment_identifier  # type:ignore
            blob.identity = blob._identity  # type:ignore
            blob.identity_corrected_solving_jumps = (
                blob._identity_corrected_solving_jumps  # type:ignore
            )
            blob.identities_corrected_closing_gaps = (
                blob._identities_corrected_closing_gaps  # type:ignore
            )
        return list_of_blobs

    @property
    def all_blobs(self) -> Iterator[Blob]:
        "A flattened view of all Blobs in the video"
        return _chain.from_iterable_of_sequences(self.blobs_in_video)

    @property
    def number_of_blobs(self) -> int:
        "Total number of Blobs in self"
        return sum(map(len, self.blobs_in_video))

    @property
    def number_of_crossing_blobs(self) -> int:
        return sum(blob.is_a_crossing for blob in self.all_blobs)

    @property
    def number_of_frames(self) -> int:
        "Number of frames in the video, equal to the number of lists of Blobs in self"
        return len(self.blobs_in_video)

    def __len__(self) -> int:
        return len(self.blobs_in_video)

    def compute_overlapping_between_subsequent_frames(self) -> None:
        """Computes overlapping between blobs in consecutive frames.

        Two blobs in consecutive frames overlap if the intersection of the list
        of pixels of both blobs is not empty.

        See Also
        --------
        :meth:`Blob.overlaps_with`
        :meth:`Blob.now_points_to`
        """
        for blob in self.all_blobs:
            blob.next = ()
            blob.previous = ()

        for blobs, blobs_next in pairwise(
            track(self.blobs_in_video, "Connecting blobs")
        ):
            for blob, blob_next in product(blobs, blobs_next):
                if blob.overlaps_with(blob_next):
                    blob.now_points_to(blob_next)

    def disconnect(self):
        for blob in self.all_blobs:
            blob.next = ()

    def reconnect(self):
        try:
            first_blob = next(self.all_blobs)
        except StopIteration:
            logging.warning("No blobs to reconnect")
            return
        if isinstance(first_blob.next, list):
            logging.info("Converting ListOfBlobs from version older than 5.2.2")
            for blob in self.all_blobs:
                blob.previous = tuple(blob.previous)
                blob.next = ()

        for blob in self.all_blobs:
            for prev_blob in blob.previous:
                prev_blob.next = prev_blob.next + (blob,)

    def set_images_for_identification(
        self,
        episodes: list[Episode],
        id_images_files: list[Path],
        id_image_size: list[int],
        resolution_reduction: float,
        n_jobs: int,
    ) -> None:
        """Computes and saves the images used to classify blobs as crossings
        and individuals and to identify the animals along the video.
        """
        if not (0 < resolution_reduction <= 1):
            logging.warning(
                "Resolution reduction should be ranged within (0, 1], current value is %s",
                resolution_reduction,
            )

        inputs = [
            (
                id_image_size[0],
                id_images_file,
                resolution_reduction,
                episode,
                self.blobs_in_video[episode.global_start : episode.global_end],
            )
            for id_images_file, episode in zip(id_images_files, episodes)
        ]

        if n_jobs == 1:
            for input in track(inputs, "Setting images for identification"):
                self.set_id_images_per_episode(input)
        else:
            with Pool(
                n_jobs,
                maxtasksperchild=3,
                initializer=setup_logging_queue,
                initargs=(LOGGING_QUEUE,),
            ) as p:
                for _ in track(
                    p.imap_unordered(self.set_id_images_per_episode, inputs),
                    "Setting images for identification",
                    len(inputs),
                ):
                    pass

        for input in inputs:
            episode, blobs_in_episode = input[3:]
            if isinstance(episode.bbox_images, BytesIO):
                episode.bbox_images.close()
            for index, blob in enumerate(chain.from_iterable(blobs_in_episode)):
                blob.id_image_index = index
                blob.episode = episode.index

    @staticmethod
    def set_id_images_per_episode(
        inputs: tuple[int, Path, float, Episode, list[list[Blob]]],
    ) -> None:
        id_image_size, file_path, res_reduct, episode, blobs_in_episode = inputs

        with (
            h5py.File(file_path, "w") as id_images_file,
            h5py.File(episode.bbox_images, "r") as bbox_images_file,
        ):
            imgs_to_save = id_images_file.create_dataset(
                "id_images",
                (sum(map(len, blobs_in_episode)), id_image_size, id_image_size),
                np.uint8,
            )

            for index, blob in enumerate(chain.from_iterable(blobs_in_episode)):
                bbox_image: np.ndarray = bbox_images_file[blob.bbox_img_id][:]  # type: ignore
                imgs_to_save[index] = blob.get_image_for_identification(
                    id_image_size, bbox_image, res_reduct
                )

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

    def update_centroid(
        self, frame_number: int, centroid_id: int, old_centroid, new_centroid
    ):
        old_centroid = tuple(old_centroid)
        new_centroid = tuple(new_centroid)
        assert len(old_centroid) == 2
        assert len(new_centroid) == 2
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

        try:
            blob_with_old_centroid = min(dist_to_old_centroid, key=lambda x: x[1])[0]
            blob_with_old_centroid.update_centroid(
                old_centroid, new_centroid, centroid_id
            )
        except ValueError:
            logging.error(
                "No blob has the centroid %s in frame %s", old_centroid, frame_number
            )

    def add_centroid(self, frame_number: int, identity: int, centroid):
        centroid = tuple(centroid)
        assert len(centroid) == 2
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

    # Deprecated methods

    @property
    @deprecated(
        version="6.0.0",
        reason="Use `max(map(len, ListOfBlobs.blobs_in_video))` instead",
    )
    def max_number_of_blobs_in_one_frame(self):
        return max(map(len, self.blobs_in_video))

    @deprecated(
        version="6.0.0",
        reason="Loading ListOfBlobs from v4 is not supported anymore",
        action="error",
    )
    def load_from_v4(self): ...

    @deprecated(
        version="6.0.0",
        reason="Use `idtrackerai.base.crossings_detection.crossing_detector._update_id_image_dataset_with_crossings` instead",
        action="error",
    )
    def update_id_image_dataset_with_crossings(self): ...

    @deprecated(
        version="6.0.0",
        reason="Use `idtrackerai.extra_tools.validator.validation_GUI.ValidationGUI.reset_session()` instead",
        action="error",
    )
    def reset_user_generated_corrections(self): ...
