import json
import logging
import pickle
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import suppress
from itertools import combinations, pairwise
from math import comb
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import h5py
import numpy as np

from . import Blob, Fragment, GlobalFragment
from .utils import (
    clean_attrs,
    deprecated,
    load_id_images,
    open_track,
    resolve_path,
    track,
)


class ListOfFragments:
    """Contains all the instances of the class :class:`Fragment`."""

    accumulable_individual_fragments: set[int]

    not_accumulable_individual_fragments: set[int]

    id_to_exclusive_roi: np.ndarray
    "Maps identities (from 0 to n_animals-1) to their exclusive ROI (-1 meaning no ROI)"

    n_animals: int

    fragments: list[Fragment]

    id_images_file_paths: list[Path]

    def __init__(
        self,
        fragments: list[Fragment],
        id_images_file_paths: list[Path],
        number_of_animals: int,
    ) -> None:
        """Initialize the ListOfFragments manually from a list of :class:`Fragment`.
        This method should only be called from :meth:`from_fragmented_blobs`. You can
        load a ListOfFragments from a file by using the method :meth:`load`.

        Parameters
        ----------
        fragments : list[Fragment]
            List of instances of the class :class:`Fragment`.
        id_images_file_paths : list[Path]
            List of paths to the files where the identification images are stored.
        number_of_animals : int
            Number of animals in the video
        """

        # Assert fragments are sorted
        assert all(i == frag.identifier for i, frag in enumerate(fragments))
        self.n_animals = number_of_animals
        self.fragments = fragments
        self.id_images_file_paths = id_images_file_paths
        self.connect_coexisting_fragments()
        self.id_to_exclusive_roi = np.full(self.n_animals, -1)

    def __iter__(self) -> Iterator[Fragment]:
        return iter(self.fragments)

    def __len__(self) -> int:
        return len(self.fragments)

    @classmethod
    def load(
        cls, file_path: Path | str, reconnect: bool = True, verbose: bool = True
    ) -> "ListOfFragments":
        """Loads a previously saved list of fragments (see :meth:`save`) from a JSON file

        Parameters
        ----------
        file_path : Path | str
            Path to the JSON file containing
        reconnect : bool, optional
            If True, the :attr:`Fragments.coexisting_individual_fragments` attribute,
            which is not saved in the JSON file, is computed and populated, by default True

        Returns
        -------
        ListOfFragments
            Loaded instance
        """
        file_path = resolve_path(file_path)
        logging.info(f"Loading ListOfFragments from {file_path}", stacklevel=2)

        if not file_path.is_file() and file_path.with_suffix(".pickle").is_file():
            # <=5.1.3 compatibility
            with file_path.with_suffix(".pickle").open("rb") as f:
                pickle.load(f).save(file_path)

        list_of_fragments = cls.__new__(cls)

        with open_track(file_path.with_suffix(".json"), verbose=verbose) as file:
            json_data: dict = json.load(file)

        list_of_fragments.accumulable_individual_fragments = set(
            json_data.get("accumulable_individual_fragments", [])
        )
        list_of_fragments.not_accumulable_individual_fragments = set(
            json_data.get("not_accumulable_individual_fragments", [])
        )
        if "number_of_animals" in json_data:
            list_of_fragments.n_animals = json_data["number_of_animals"]
        if "n_animals" in json_data:
            list_of_fragments.n_animals = json_data["n_animals"]

        list_of_fragments.id_images_file_paths = list(
            map(Path, json_data["id_images_file_paths"])
        )

        list_of_fragments.fragments = [
            Fragment.from_json(frag_data) for frag_data in json_data["fragments"]
        ]

        for fragment in list_of_fragments:
            if (
                fragment.identifier
                in list_of_fragments.accumulable_individual_fragments
            ):
                fragment.accumulable = True
            elif (
                fragment.identifier
                in list_of_fragments.not_accumulable_individual_fragments
            ):
                fragment.accumulable = False

        if reconnect:
            list_of_fragments.connect_coexisting_fragments(verbose=verbose)

        with suppress(AttributeError):
            list_of_fragments.id_to_exclusive_roi = np.asarray(
                json_data.get(
                    "id_to_exclusive_roi", np.full(list_of_fragments.n_animals, -1)
                )
            )

        new_location = file_path.parent.parent / "identification_images"
        if new_location != list_of_fragments.id_images_file_paths[0].parent and all(
            (new_location / file.name).is_file()
            for file in list_of_fragments.id_images_file_paths
        ):
            logging.info("Updating List Of Fragments id_images_file_paths")
            list_of_fragments.id_images_file_paths = [
                new_location / file.name
                for file in list_of_fragments.id_images_file_paths
            ]

        return list_of_fragments

    @classmethod
    def from_fragmented_blobs(
        cls,
        all_blobs: Iterable[Blob],
        number_of_animals: int,
        id_images_file_paths: list[Path],
    ) -> "ListOfFragments":
        """Generate a list of instances of :class:`Fragment` collecting
        all the fragments in the video.

        Parameters
        ----------
        all_blobs : Iterable[Blob]
            An iterable of the blob objects which have to be sorted by
            :attr:`Blob.frame_number`, for example :attr:`ListOfBlobs.all_blobs`.
        number_of_animals : int
            Number of animals to track as defined by the user
        id_images_file_paths : list[Path]
            List of paths to the files where the identification images are stored.

        Returns
        -------
        ListOfFragments
            The List of Fragments computed from the blobs.
        """

        fragments: list[Fragment] = []
        used_fragment_identifiers: set[int] = set()

        logging.info("Creating list of fragments")
        for blob in all_blobs:
            current_fragment_identifier = blob.fragment_identifier
            if current_fragment_identifier in used_fragment_identifiers:
                continue
            images = [blob.id_image_index]
            centroids = [blob.centroid]
            episodes = [blob.episode]
            start = blob.frame_number
            exclusive_roi = blob.exclusive_roi
            current = blob

            while (
                current.n_next > 0
                and current.next[0].fragment_identifier == current_fragment_identifier
            ):
                current = current.next[0]
                images.append(current.id_image_index)
                centroids.append(current.centroid)
                episodes.append(current.episode)

            end = current.frame_number

            fragment = Fragment(
                current_fragment_identifier,
                start,
                end + 1,  # it is not inclusive
                images,
                centroids,
                episodes,
                blob.is_an_individual,
                exclusive_roi,
            )
            used_fragment_identifiers.add(current_fragment_identifier)
            fragments.append(fragment)
        return cls(fragments, id_images_file_paths, number_of_animals)

    def save(self, file_path: Path | str) -> None:
        """Save the list of fragments into a JSON file

        Parameters
        ----------
        file_path : Path | str
            Path where the instance of the object will be stored.
        """
        file_path = resolve_path(file_path)
        if file_path.is_dir():
            file_path /= "list_of_fragments.json"
        logging.info(f"Saving ListOfFragments as {file_path}", stacklevel=2)
        file_path.parent.mkdir(exist_ok=True)

        with open_track(file_path, "w") as file:
            json.dump(self, file, cls=FragmentsEncoder, indent=4)

    @property
    def number_of_fragments(self) -> int:
        return len(self.fragments)

    @property
    def individual_fragments(self) -> Generator[Fragment, None, None]:
        return (frag for frag in self if frag.is_an_individual)

    # TODO: if the resume feature is not active, this does not make sense|
    def reset(self, roll_back_to: Literal["fragmentation", "accumulation"]) -> None:
        """Resets all the fragment to a given processing step.

        Parameters
        ----------
        roll_back_to : str
            Name of the step at which the fragments should be reset.
            It can be 'fragmentation', 'accumulation' or

        See Also
        --------
        :meth:`Fragment.reset`
        """
        logging.info(f"Resetting ListOfFragments to '{roll_back_to}'", stacklevel=2)
        for fragment in self:
            fragment.reset(roll_back_to, self.n_animals)

    @property
    def n_images_in_global_fragments(self) -> int:
        """Number of images available in global fragments
        (without repetitions)"""
        return sum(
            frag.n_images
            for frag in self
            if frag.identifier in self.accumulable_individual_fragments
        )

    @property
    def ratio_of_images_used_for_training(self) -> float:
        """Returns the ratio of images used for training over the number of
        available images"""
        return (
            sum(fragment.n_images for fragment in self if fragment.used_for_training)
            / self.n_images_in_global_fragments
        )

    def build_exclusive_rois(self) -> dict[str, set[int]]:
        """Builds `id_to_exclusive_roi` and returns a more readable version
        intended to be saved in Session.identities_groups"""

        # build id_to_exclusive_roi
        for fragment in self:
            if fragment.temporary_id is not None:
                self.id_to_exclusive_roi[fragment.temporary_id] = fragment.exclusive_roi

        # build identity groups to save in Session
        id_groups: dict[str, set] = {}
        for id, roi in enumerate(self.id_to_exclusive_roi):
            if roi == -1:
                continue
            roi_name = f"Region_{roi}"
            if roi_name in id_groups:
                id_groups[roi_name].add(id + 1)
            else:
                id_groups[roi_name] = {id + 1}
        if id_groups:
            logging.info("Identity groups by exclusive ROIs:\n%s", pformat(id_groups))

        return id_groups

    def compute_P2_vectors(self) -> None:
        """Computes the P2_vector associated to every individual fragment. See
        :meth:`Fragment.compute_P2_vector`
        """
        for fragment in self.individual_fragments:
            fragment.compute_P2_vector(self.n_animals)

    def get_fragments_to_identify(self) -> Iterator[Fragment]:
        """Yields all the :class:`Fragment` instances left non identified after
        the cascade of training and identification protocols sorted by the
        certainty computed with P2. See :attr:Fragment.certainty_P2`

        Yields
        ------
        Iterator[Fragment]
            Non identified Fragments sorted by P2 certainty
        """
        frags_to_identity = [
            frag for frag in self.individual_fragments if frag.identity is None
        ]
        while frags_to_identity:
            yield max(frags_to_identity, key=lambda frag: frag.certainty_P2)
            frags_to_identity = [
                frag for frag in self.individual_fragments if frag.identity is None
            ]

    def update_id_images_dataset(self) -> None:
        """Updates the identification images files with the identity assigned
        to each fragment during the tracking process.
        """
        logging.info("Updating identities in identification images files")

        identities = []
        for path in self.id_images_file_paths:
            with h5py.File(path, "r") as file:
                identities.append(np.full(len(file["id_images"]), 0))  # type: ignore

        for fragment in self:
            if fragment.used_for_training:
                for image, episode in fragment.image_locations:
                    identities[episode][image] = fragment.identity

        for path, identities_in_episode in zip(self.id_images_file_paths, identities):
            try:
                with h5py.File(path, "r+") as file:
                    dataset = file.require_dataset(
                        "identities", shape=len(identities_in_episode), dtype=int
                    )
                    dataset[:] = identities_in_episode
            except BlockingIOError as exc:
                # Some MacOS crash with
                # BlockingIOError: [Errno 35] Unable to open file (unable to lock file, errno = 35, error message = 'Resource temporarily unavailable')
                logging.error(f"Failed to update identities in {path} due to: {exc}")

    def get_ordered_list_of_fragments(
        self, scope: Literal["to_the_past", "to_the_future"], specific_frame: int
    ) -> list[Fragment]:
        """Sorts the fragments starting from the frame number
        `first_frame_first_global_fragment`. According to `scope` the sorting
        is done either "to the future" of "to the past" with respect to
        `first_frame_first_global_fragment`

        Parameters
        ----------
        scope : str
            either "to_the_past" or "to_the_future"
        first_frame_first_global_fragment : int
            frame number corresponding to the first frame in which all the
            individual fragments coexist in the first global fragment using
            in an iteration of the fingerprint protocol cascade

        Returns
        -------
        list
            list of sorted fragments

        """
        if scope == "to_the_past":
            fragments_to_the_past = filter(
                lambda frag: frag.end_frame <= specific_frame, self.fragments
            )
            return sorted(
                fragments_to_the_past, key=lambda x: x.end_frame, reverse=True
            )
        if scope == "to_the_future":
            fragments_to_the_future = filter(
                lambda frag: frag.start_frame >= specific_frame, self.fragments
            )
            return sorted(fragments_to_the_future, key=lambda x: x.start_frame)
        raise ValueError(scope)

    def load_images_in_memory(
        self, condition: Callable[[Fragment], bool] | None = None
    ):
        """Loads Fragment's images in memory from id_images_file_paths.
        It only takes into account the fragments satisfying the given condition.
        Used outside idtracker.ai"""
        image_locations = []
        fragments = list(filter(condition, self)) if condition is not None else self
        for frag in fragments:
            image_locations += frag.image_locations
        all_images = load_id_images(self.id_images_file_paths, image_locations)
        counter = 0
        for frag in fragments:
            frag.loaded_images = all_images[counter : counter + frag.n_images]
            counter += frag.n_images
        assert counter == len(all_images)

    def connect_coexisting_fragments(self, verbose=True) -> None:
        "Populates :attr:`Fragment.coexisting_individual_fragments`"
        for fragment in self:
            fragment.coexisting_individual_fragments = []

        # If fragments are sorted by starting frame, we have a nice optimization
        if all(
            frag_A.start_frame <= frag_B.start_frame and frag_A.identifier == i
            for i, (frag_A, frag_B) in enumerate(pairwise(self.fragments))
        ):
            for fragment_A in track(
                self.fragments, "Connecting coexisting fragments", verbose=verbose
            ):
                for fragment_B in self.fragments[fragment_A.identifier + 1 :]:
                    if fragment_A.coexist_with(fragment_B):
                        if fragment_A.is_an_individual:
                            fragment_B.coexisting_individual_fragments.append(fragment_A)  # type: ignore
                        if fragment_B.is_an_individual:
                            fragment_A.coexisting_individual_fragments.append(fragment_B)  # type: ignore
                    if fragment_B.start_frame > fragment_A.end_frame:
                        break
            return

        # brute force
        logging.warning("List of Fragments is not sorted")
        for fragment_A, fragment_B in track(
            combinations(self.fragments, 2),
            "Connecting coexisting fragments",
            comb(len(self.fragments), 2),
            verbose=verbose,
        ):
            if fragment_A.coexist_with(fragment_B):
                if fragment_A.is_an_individual:
                    fragment_B.coexisting_individual_fragments.append(fragment_A)  # type: ignore
                if fragment_B.is_an_individual:
                    fragment_A.coexisting_individual_fragments.append(fragment_B)  # type: ignore

    def manage_accumulable_non_accumulable_fragments(
        self,
        accumulable_global_fragments: list[GlobalFragment],
        non_accumulable_global_fragments: list[GlobalFragment],
    ):
        """Gets the unique identifiers associated to individual fragments that
        can be accumulated.

        Parameters
        ----------
        list_of_global_fragments : :class:`ListOfGlobalFragments`
            Object collecting the global fragment objects (instances of the
            class :class:`GlobalFragment`) detected in the
            entire video.

        """
        self.accumulable_individual_fragments = {
            identifier
            for glob_frag in accumulable_global_fragments
            for identifier in glob_frag.fragments_identifiers
        }
        self.not_accumulable_individual_fragments = {
            identifier
            for glob_frag in non_accumulable_global_fragments
            for identifier in glob_frag.fragments_identifiers
        } - self.accumulable_individual_fragments

        for fragment in self:
            if fragment.identifier in self.accumulable_individual_fragments:
                fragment.accumulable = True
            elif fragment.identifier in self.not_accumulable_individual_fragments:
                fragment.accumulable = False

    @property
    def number_of_crossing_fragments(self) -> int:
        return sum(fragment.is_a_crossing for fragment in self)

    @property
    def number_of_individual_fragments(self) -> int:
        return sum(1 for _ in self.individual_fragments)

    def get_stats(self) -> dict[str, int]:
        """Collects the following counters from the fragments.

        * fragments
        * crossing_fragments
        * individual_fragments
        * individual_fragments_not_in_a_global_fragment
        * accumulable_individual_fragments
        * not_accumulable_individual_fragments
        * globally_accumulated_individual_fragments
        * partially_accumulated_individual_fragments
        * blobs
        * crossing_blobs
        * individual_blobs
        * individual_blobs_not_in_a_global_fragment
        * accumulable_individual_blobs
        * not_accumulable_individual_blobs
        * globally_accumulated_individual_blobs
        * partially_accumulated_individual_blobs

        Returns
        -------
        dict
            Dictionary with the counters mentioned above

        """

        stats: dict[str, int] = {
            "fragments": self.number_of_fragments,
            "crossing_fragments": self.number_of_crossing_fragments,
            "individual_fragments": self.number_of_individual_fragments,
            "individual_fragments_not_in_a_global_fragment": sum(
                not frag.is_in_a_global_fragment for frag in self.individual_fragments
            ),
            "accumulable_individual_fragments": len(
                self.accumulable_individual_fragments
            ),
            "not_accumulable_individual_fragments": len(
                self.not_accumulable_individual_fragments
            ),
            "globally_accumulated_individual_fragments": sum(
                frag.accumulated_globally for frag in self.individual_fragments
            ),
            "partially_accumulated_individual_fragments": sum(
                frag.accumulated_partially for frag in self.individual_fragments
            ),
            "blobs": sum(frag.n_images for frag in self),
            "crossing_blobs": sum(frag.is_a_crossing * frag.n_images for frag in self),
            "individual_blobs": sum(
                frag.n_images for frag in self.individual_fragments
            ),
            "individual_blobs_not_in_a_global_fragment": sum(
                not frag.is_in_a_global_fragment * frag.n_images
                for frag in self.individual_fragments
            ),
            "accumulable_individual_blobs": sum(
                bool(frag.accumulable) * frag.n_images for frag in self
            ),
            "not_accumulable_individual_blobs": sum(
                (not frag.accumulable) * frag.n_images
                for frag in self
                if frag.accumulable is not None
            ),
            "globally_accumulated_individual_blobs": sum(
                frag.accumulated_globally * frag.n_images
                for frag in self.individual_fragments
            ),
            "partially_accumulated_individual_blobs": sum(
                frag.accumulated_partially * frag.n_images
                for frag in self.individual_fragments
            ),
        }

        log = "Final statistics:"
        for key, value in stats.items():
            log += f"\n  {value} {key.replace('_', ' ')}"
        logging.info(log)

        return stats

    def update_blobs(self, all_blobs: Iterable[Blob]) -> None:
        """Updates the blobs objects generated from the video with the
        attributes computed for each fragment

        Parameters
        ----------
        all_blobs : Iterable[Blob]
            All blobs to be updated
        """
        logging.info("Updating list of blobs from list of fragments")
        for blob in all_blobs:
            fragment = self.fragments[blob.fragment_identifier]
            blob.identity = fragment.identity
            blob.identity_corrected_solving_jumps = (
                fragment.identity_corrected_solving_jumps
            )
            blob.is_an_individual = fragment.is_an_individual
            if fragment.forced_crossing:
                blob.forced_crossing = True

    # Deprecated methods

    @deprecated(
        version="6.0.0",
        reason="Use :meth:`get_fragments_to_identify` instead",
        action="error",
    )
    def get_next_fragment_to_identify(self): ...

    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def get_number_of_unidentified_individual_fragments(): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def ratio_of_images_used_for_pretraining(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_individual_fragments_not_in_a_glob_fragment(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_accumulable_individual_fragments(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_not_accumulable_individual_fragments(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_blobs(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_crossing_blobs(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_individual_blobs(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_individual_blobs_not_in_a_global_fragment(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def fragments_not_accumulated(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_globally_accumulated_individual_fragments(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_partially_accumulated_individual_fragments(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_accumulable_individual_blobs(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_not_accumulable_individual_blobs(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_globally_accumulated_individual_blobs(self): ...
    @property
    @deprecated(
        version="6.0.0",
        reason="See :meth:`get_stats` to get the same information",
        action="error",
    )
    def number_of_partially_accumulated_individual_blobs(self): ...


class FragmentsEncoder(json.JSONEncoder):
    def default(self, o):
        match o:
            case Path():
                return str(o)

            case ListOfFragments():
                serial = o.__dict__.copy()
                serial["id_to_exclusive_roi"] = (
                    f"NotString{serial.get('id_to_exclusive_roi', np.array(())).tolist()}"
                )
                serial["accumulable_individual_fragments"] = (
                    f"NotString{list(serial.get('accumulable_individual_fragments', {}))}"
                )
                serial["not_accumulable_individual_fragments"] = (
                    f"NotString{list(serial.get('not_accumulable_individual_fragments', {}))}"
                )
                return serial

            case Fragment():
                clean_attrs(o)
                serial = o.__getstate__()

                serial["episodes"] = f"NotString{compress(o.episodes).tolist()}"

                for key in (
                    "frame_by_frame_velocity",
                    "start_position",
                    "end_position",
                ):
                    if key in serial:
                        serial[key] = f"NotString{np.round(serial[key], 2).tolist()}"

                for key in ("images", "P1_vector", "P2_vector", "ambiguous_identities"):
                    if key in serial:
                        serial[key] = f"NotString{serial[key].tolist()}"

                return serial
            case np.integer():
                return int(o)
            case np.bool_():
                return bool(o)
            case np.floating():
                return float(o)
            case _:
                return super().default(o)

    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
        for encoded in super().iterencode(o, _one_shot):
            if encoded.startswith('"NotString'):
                # remove first and final '"NoIndent..."' and remove indents,
                yield encoded[10:-1]
            else:
                yield encoded


def compress(arr: np.ndarray) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Compresses an integer 1D array by finding its repetitions.

    [0,0,0,1,1,2,2,2] -> [[0,1,2], [3,2,3]]"""
    if len(arr) == 0:
        return np.array([[], []], dtype=int)
    changing_indices: np.ndarray = np.diff(arr, prepend=-1, append=-1) != 0
    values = arr[changing_indices[:-1]]
    repetitions = np.diff((~changing_indices).cumsum()[changing_indices]) + 1
    return np.asarray((values, repetitions))
