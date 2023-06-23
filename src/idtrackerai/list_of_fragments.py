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
import json
import logging
import pickle
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Iterable, Literal

import h5py
import numpy as np

from . import Blob, Fragment, GlobalFragment
from .utils import clean_attrs, load_id_images, resolve_path, track


class ListOfFragments:
    """Contains all the instances of the class :class:`fragment.Fragment`.

    Parameters
    ----------
    fragments : list
        List of instances of the class :class:`fragment.Fragment`.
    id_images_file_paths : list
        List of strings with the paths to the files where the identification
        images are stored.
    """

    accumulable_individual_fragments: set[int]
    not_accumulable_individual_fragments: set[int]

    def __init__(
        self,
        fragments: list[Fragment],
        id_images_file_paths: list[Path],
        number_of_animals: int,
    ):
        # Assert fragments are sorted
        for i, fragment in enumerate(fragments):
            assert i == fragment.identifier
        self.number_of_animals = number_of_animals
        self.fragments = fragments
        self.id_images_file_paths = id_images_file_paths
        self.connect_coexisting_fragments()

    @property
    def number_of_fragments(self):
        return len(self.fragments)

    @property
    def individual_fragments(self):
        return (frag for frag in self.fragments if frag.is_an_individual)

    # TODO: if the resume feature is not active, this does not make sense|
    def reset(self, roll_back_to: Literal["fragmentation", "accumulation"]):
        """Resets all the fragment to a given processing step.

        Parameters
        ----------
        roll_back_to : str
            Name of the step at which the fragments should be reset.
            It can be 'fragmentation', 'pretraining', 'accumulation' or
            'assignment'

        See Also
        --------
        :meth:`fragment.Fragment.reset`
        """
        logging.info(f"Resetting ListOfFragments to '{roll_back_to}'", stacklevel=3)
        for fragment in self.fragments:
            fragment.reset(roll_back_to, self.number_of_animals)

    # TODO: maybe this should go to the accumulator manager
    def get_images_from_fragments_to_assign(self):
        """Take all the fragments that have not been used to train the idCNN
        and that are associated with an individual, and concatenates their
        images in order to feed them to the identification network.

        Returns
        -------
        ndarray
            [number_of_images, height, width, number_of_channels]
        """
        images: list[tuple[int, int]] = []
        for fragment in self.individual_fragments:
            if not fragment.used_for_training:
                images += fragment.image_locations

        logging.info(
            f"Number of images to identify non-accumulated fragments: {len(images)}"
        )
        return load_id_images(self.id_images_file_paths, images)

    # TODO: The following methods could be properties.
    # TODO: The following methods depend on the identification strategy.
    def compute_number_of_unique_images_used_for_pretraining(self):
        """Returns the number of images used for pretraining
        (without repetitions)

        Returns
        -------
        int
            Number of images used in pretraining
        """
        return sum(
            fragment.number_of_images
            for fragment in self.fragments
            if fragment.used_for_pretraining
        )

    def compute_number_of_unique_images_used_for_training(self):
        """Returns the number of images used for training
        (without repetitions)

        Returns
        -------
        int
            Number of images used in training
        """
        return sum(
            fragment.number_of_images
            for fragment in self.fragments
            if fragment.used_for_training
        )

    @property
    def number_of_images_in_global_fragments(self) -> int:
        """Number of images available in global fragments
        (without repetitions)"""
        return sum(
            fragment.number_of_images
            for fragment in self.fragments
            if fragment.identifier in self.accumulable_individual_fragments
        )

    def compute_ratio_of_images_used_for_pretraining(self):
        """Returns the ratio of images used for pretraining over the number of
        available images

        Returns
        -------
        float
            Ratio of images used for pretraining
        """
        return (
            self.compute_number_of_unique_images_used_for_pretraining()
            / self.number_of_images_in_global_fragments
        )

    def compute_ratio_of_images_used_for_training(self):
        """Returns the ratio of images used for training over the number of
        available images

        Returns
        -------
        float
            Ratio of images used for training
        """
        return (
            self.compute_number_of_unique_images_used_for_training()
            / self.number_of_images_in_global_fragments
        )

    def compute_P2_vectors(self):
        """Computes the P2_vector associated to every individual fragment. See
        :meth:`fragment.Fragment.compute_P2_vector`
        """
        for fragment in self.individual_fragments:
            fragment.compute_P2_vector(self.number_of_animals)

    def get_number_of_unidentified_individual_fragments(self):
        """Returns the number of individual fragments that have not been
        identified during the fingerprint protocols cascade

        Returns
        -------
        int
            number of non-identified individual fragments
        """
        return sum(
            frag.is_an_individual and not frag.used_for_training
            for frag in self.fragments
        )

    def get_next_fragment_to_identify(self) -> Fragment | None:
        """Returns the next fragment to be identified after the cascade of
        training and identitication protocols by sorting according to the
        certainty computed with P2. See :attr:fragment.Fragment.certainty_P2`

        Returns
        -------
        :class:`fragment.Fragment`
            An instance of the class :class:`fragment.Fragment`
        """
        try:
            return max(
                filter(
                    lambda frag: frag.is_an_individual
                    and frag.assigned_identities[0] is None,
                    self.fragments,
                ),
                key=lambda frag: frag.certainty_P2,
            )
        except ValueError:
            return None

    def update_id_images_dataset(self):
        """Updates the identification images files with the identity assigned
        to each fragment during the tracking process.
        """
        logging.info("Updating identities in identification images files")

        identities = []
        for path in self.id_images_file_paths:
            with h5py.File(path, "r") as file:
                identities.append(np.full(file["id_images"].shape[0], 0))  # type: ignore

        for fragment in self.fragments:
            if fragment.used_for_training:
                for image, episode in fragment.image_locations:
                    identities[episode][image] = fragment.identity

        for path, identities_in_episode in zip(self.id_images_file_paths, identities):
            with h5py.File(path, "r+") as file:
                dataset = file.require_dataset(
                    "identities", shape=len(identities_in_episode), dtype=int
                )
                dataset[:] = identities_in_episode

    def get_ordered_list_of_fragments(
        self, scope: str, specific_frame: int
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

    def save(self, path: Path | str):
        """Save an instance of the object in disk,

        Parameters
        ----------
        fragments_path : str
            Path where the instance of the object will be stored.
        """
        path = resolve_path(path)
        logging.info(f"Saving ListOfFragments as {path}")
        path.parent.mkdir(exist_ok=True)

        json.dump(self, path.open("w"), cls=FragmentsEncoder, indent=4)

    @classmethod
    def load(cls, path: Path | str, reconnect=True) -> "ListOfFragments":
        """Loads a previously saved (see :meth:`save`) from the path
        `path_to_load`
        """
        path = resolve_path(path)
        logging.info(f"Loading ListOfFragments from {path}")

        if not path.is_file() and path.with_suffix(".pickle").is_file():
            # <=5.1.3 compatibility
            pickle.load(path.with_suffix(".pickle").open("rb")).save(path)

        list_of_fragments = cls.__new__(cls)
        json_data = json.load(path.with_suffix(".json").open("r"))

        list_of_fragments.accumulable_individual_fragments = set(
            json_data.get("accumulable_individual_fragments", [])
        )
        list_of_fragments.not_accumulable_individual_fragments = set(
            json_data.get("not_accumulable_individual_fragments", [])
        )
        if "number_of_animals" in json_data:
            list_of_fragments.number_of_animals = json_data["number_of_animals"]
        list_of_fragments.id_images_file_paths = list(
            map(Path, json_data["id_images_file_paths"])
        )

        list_of_fragments.fragments = [
            Fragment.from_json(frag_data) for frag_data in json_data["fragments"]
        ]

        for fragment in list_of_fragments.fragments:
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
            list_of_fragments.connect_coexisting_fragments()

        return list_of_fragments

    def connect_coexisting_fragments(self):
        logging.info("Connecting coexisting individual fragments")
        # Make it N (not N²) with, maybe, sets (not lists)
        for fragment in self.fragments:
            fragment.coexisting_individual_fragments = []

        for fragment_A, fragment_B in track(
            combinations(self.fragments, 2),
            "Connecting coexisting fragments",
            comb(len(self.fragments), 2),
        ):
            if fragment_A.coexist_with(fragment_B):
                if fragment_A.is_an_individual:
                    fragment_B.coexisting_individual_fragments.append(fragment_A)
                if fragment_B.is_an_individual:
                    fragment_A.coexisting_individual_fragments.append(fragment_B)

    def manage_accumulable_non_accumulable_fragments(
        self,
        accumulable_global_fragments: list[GlobalFragment],
        non_accumulable_global_fragments: list[GlobalFragment],
    ):
        """Gets the unique identifiers associated to individual fragments that
        can be accumulated.

        Parameters
        ----------
        list_of_global_fragments : :class:`list_of_global_fragments.ListOfGlobalFragments`
            Object collecting the global fragment objects (instances of the
            class :class:`globalfragment.GlobalFragment`) detected in the
            entire video.

        """
        self.accumulable_individual_fragments = {
            identifier
            for glob_frag in accumulable_global_fragments
            for identifier in glob_frag.individual_fragments_identifiers
        }
        self.not_accumulable_individual_fragments = {
            identifier
            for glob_frag in non_accumulable_global_fragments
            for identifier in glob_frag.individual_fragments_identifiers
        } - self.accumulable_individual_fragments

        for fragment in self.fragments:
            if fragment.identifier in self.accumulable_individual_fragments:
                fragment.accumulable = True
            elif fragment.identifier in self.not_accumulable_individual_fragments:
                fragment.accumulable = False

    @property
    def number_of_crossing_fragments(self) -> int:
        return sum(fragment.is_a_crossing for fragment in self.fragments)

    @property
    def number_of_individual_fragments(self) -> int:
        return sum(1 for _ in self.individual_fragments)

    @property
    def number_of_individual_fragments_not_in_a_glob_fragment(self) -> int:
        return sum(
            not fragment.is_in_a_global_fragment
            for fragment in self.individual_fragments
        )

    @property
    def number_of_accumulable_individual_fragments(self) -> int:
        return len(self.accumulable_individual_fragments)

    @property
    def number_of_not_accumulable_individual_fragments(self) -> int:
        return len(self.not_accumulable_individual_fragments)

    @property
    def number_of_blobs(self) -> int:
        return sum(fragment.number_of_images for fragment in self.fragments)

    @property
    def number_of_crossing_blobs(self) -> int:
        return sum(
            fragment.is_a_crossing * fragment.number_of_images
            for fragment in self.fragments
        )

    @property
    def number_of_individual_blobs(self) -> int:
        return sum(fragment.number_of_images for fragment in self.individual_fragments)

    @property
    def number_of_individual_blobs_not_in_a_global_fragment(self) -> int:
        return sum(
            not fragment.is_in_a_global_fragment * fragment.number_of_images
            for fragment in self.individual_fragments
        )

    @property
    def fragments_not_accumulated(self) -> set[int]:
        return self.accumulable_individual_fragments & {
            fragment.identifier
            for fragment in self.fragments
            if not fragment.used_for_training
        }

    @property
    def number_of_globally_accumulated_individual_fragments(self) -> int:
        return sum(
            fragment.accumulated_globally for fragment in self.individual_fragments
        )

    @property
    def number_of_partially_accumulated_individual_fragments(self) -> int:
        return sum(
            fragment.accumulated_partially for fragment in self.individual_fragments
        )

    @property
    def number_of_accumulable_individual_blobs(self) -> int:
        return sum(
            bool(fragment.accumulable) * fragment.number_of_images
            for fragment in self.fragments
        )

    @property
    def number_of_not_accumulable_individual_blobs(self) -> int:
        return sum(
            (not fragment.accumulable) * fragment.number_of_images
            for fragment in self.fragments
            if fragment.accumulable is not None
        )

    @property
    def number_of_globally_accumulated_individual_blobs(self) -> int:
        return sum(
            fragment.accumulated_globally * fragment.number_of_images
            for fragment in self.individual_fragments
        )

    @property
    def number_of_partially_accumulated_individual_blobs(self) -> int:
        return sum(
            fragment.accumulated_partially * fragment.number_of_images
            for fragment in self.individual_fragments
        )

    def get_stats(self) -> dict[str, Any]:
        """Collects the following counters from the fragments.

        * number_of_fragments
        * number_of_crossing_fragments
        * number_of_individual_fragments
        * number_of_individual_fragments_not_in_a_glob_fragment
        * number_of_accumulable_individual_fragments
        * number_of_not_accumulable_individual_fragments
        * number_of_accumulated_individual_fragments
        * number_of_globally_accumulated_individual_fragments
        * number_of_partially_accumulated_individual_fragments
        * number_of_blobs
        * number_of_crossing_blobs
        * number_of_individual_blobs
        * number_of_individual_blobs_not_in_a_global_fragment
        * number_of_accumulable_individual_blobs
        * number_of_not_accumulable_individual_blobs
        * number_of_accumulated_individual_blobs
        * number_of_globally_accumulated_individual_blobs
        * number_of_partially_accumulated_individual_blobs

        Returns
        -------
        dict
            Dictionary with the counters mentioned above

        """

        stats: dict[str, Any] = {
            "fragments": self.number_of_fragments,
            "crossing_fragments": self.number_of_crossing_fragments,
            "individual_fragments": self.number_of_individual_fragments,
            "individual_fragments_not_in_a_global_fragment": (
                self.number_of_individual_fragments_not_in_a_glob_fragment
            ),
            "accumulable_individual_fragments": (
                self.number_of_accumulable_individual_fragments
            ),
            "not_accumulable_individual_fragments": (
                self.number_of_not_accumulable_individual_fragments
            ),
            "globally_accumulated_individual_fragments": (
                self.number_of_globally_accumulated_individual_fragments
            ),
            "partially_accumulated_individual_fragments": (
                self.number_of_partially_accumulated_individual_fragments
            ),
            "blobs": self.number_of_blobs,
            "crossing_blobs": self.number_of_crossing_blobs,
            "individual_blobs": self.number_of_individual_blobs,
            "individual_blobs_not_in_a_global_fragment": (
                self.number_of_individual_blobs_not_in_a_global_fragment
            ),
            "accumulable_individual_blobs": self.number_of_accumulable_individual_blobs,
            "not_accumulable_individual_blobs": (
                self.number_of_not_accumulable_individual_blobs
            ),
            "globally_accumulated_individual_blobs": (
                self.number_of_globally_accumulated_individual_blobs
            ),
            "partially_accumulated_individual_blobs": (
                self.number_of_partially_accumulated_individual_blobs
            ),
        }

        log = "Final statistics:"
        for key, value in stats.items():
            log += f"\n  {value} {key.replace('_', ' ')}"
        logging.info(log)

        return stats

    @classmethod
    def from_fragmented_blobs(
        cls,
        all_blobs: Iterable[Blob],
        number_of_animals: int,
        id_images_file_paths: list[Path],
    ) -> "ListOfFragments":
        """Generate a list of instances of :class:`fragment.Fragment` collecting
        all the fragments in the video.

        Parameters
        ----------
        blobs_in_video : list
            list of the blob objects (see class :class:`blob.Blob`) generated
            from the blobs segmented in the video
        number_of_animals : int
            Number of animals to track as defined by the user

        Returns
        -------
        list
            list of instances of :class:`fragment.Fragment`

        """
        fragments: list[Fragment] = []
        used_fragment_identifiers: set[int] = set()

        logging.info("Creating list of fragments")
        for blob in all_blobs:
            current_fragment_identifier = blob.fragment_identifier
            if current_fragment_identifier not in used_fragment_identifiers:
                images = [blob.id_image_index]
                centroids = [blob.centroid]
                episodes = [blob.episode]
                start = blob.frame_number
                current = blob

                while (
                    current.n_next > 0
                    and current.next[0].fragment_identifier
                    == current_fragment_identifier
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
                )
                used_fragment_identifiers.add(current_fragment_identifier)
                fragments.append(fragment)
        return cls(fragments, id_images_file_paths, number_of_animals)

    def update_blobs(self, all_blobs: Iterable[Blob]):
        """Updates the blobs objects generated from the video with the
        attributes computed for each fragment

        Parameters
        ----------
        fragments : list
            List of all the fragments

        See Also
        --------
        :meth:`blob.Blob.compute_fragment_identifier_and_blob_index`

        """
        logging.info("Updating list of blobs from list of fragments")
        for blob in all_blobs:
            fragment = self.fragments[blob.fragment_identifier]
            blob.identity = fragment.identity
            blob.identity_corrected_solving_jumps = (
                fragment.identity_corrected_solving_jumps
            )
            blob.user_generated_identity = fragment.user_generated_identity
            blob.is_an_individual = fragment.is_an_individual
            if fragment.forced_crossing:
                blob.forced_crossing = True


class FragmentsEncoder(json.JSONEncoder):
    def default(self, obj):
        match obj:
            case Path():
                return str(obj)

            case ListOfFragments():
                serial = obj.__dict__.copy()
                serial["accumulable_individual_fragments"] = (
                    f"NotString{json.dumps(list(serial.get('accumulable_individual_fragments',{})))}"
                )
                serial["not_accumulable_individual_fragments"] = (
                    f"NotString{json.dumps(list(serial.get('not_accumulable_individual_fragments',{})))}"
                )
                return serial

            case Fragment():
                clean_attrs(obj)
                serial = obj.__dict__.copy()
                serial.pop("coexisting_individual_fragments", None)
                serial.pop("centroids", None)  # v5.1.3 compatibility
                serial.pop("accumulable", None)
                serial["images"] = "NotString" + json.dumps(obj.images)
                if len(set(obj.episodes)) == 1:
                    # compress when all images are in the same episode
                    serial["episodes"] = f"NotString{[obj.episodes[0]]}"
                else:
                    serial["episodes"] = "NotString" + json.dumps(obj.episodes)
                if "frame_by_frame_velocity" in serial:
                    serial["frame_by_frame_velocity"] = "NotString" + json.dumps(
                        np.round(obj.frame_by_frame_velocity, 2).tolist()
                    )
                if "start_position" in serial:
                    serial["start_position"] = "NotString" + json.dumps(
                        np.round(obj.start_position, 2).tolist()
                    )
                if "end_position" in serial:
                    serial["end_position"] = "NotString" + json.dumps(
                        np.round(obj.end_position, 2).tolist()
                    )
                for key in ("P1_vector", "P2_vector", "ambiguous_identities"):
                    if key in serial:
                        serial[key] = "NotString" + json.dumps(serial[key].tolist())

                return serial
            case np.integer():
                return int(obj)
            case np.bool_():
                return bool(obj)
            case np.floating():
                return float(obj)
            case _:
                return super().default(obj)

    def iterencode(self, obj, **kwargs):
        for encoded in super().iterencode(obj, **kwargs):
            if encoded.startswith('"NotString'):
                # remove first and final '"NoIndent..."' and remove indents,
                yield encoded[10:-1]
            else:
                yield encoded
