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
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from rich.progress import track

from idtrackerai import Blob, Fragment, GlobalFragment
from idtrackerai.utils import load_id_images, resolve_path


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

    def __init__(self, fragments: list[Fragment], id_images_file_paths: list[Path]):
        # Assert fragments are sorted
        for i, fragment in enumerate(fragments):
            assert i == fragment.identifier

        self.fragments = fragments
        self.id_images_file_paths = id_images_file_paths
        self.connect_coexisting_fragments()

    @property
    def number_of_fragments(self):
        return len(self.fragments)

    # TODO: if the resume feature is not active, this does not make sense|
    def reset(self, roll_back_to):
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
        logging.info(f"Resetting ListOfFragments to '{roll_back_to}'")
        for fragment in self.fragments:
            fragment.reset(roll_back_to)

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
        images_lists = [
            list(zip(fragment.images, fragment.episodes))
            for fragment in self.fragments
            if not fragment.used_for_training and fragment.is_an_individual
        ]
        images = [image for images in images_lists for image in images]
        logging.info(
            "Number of images to identify non-accumulated " f"fragments: {len(images)}"
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
        for fragment in self.fragments:
            if fragment.is_an_individual:
                fragment.compute_P2_vector()

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
                (
                    fragment
                    for fragment in self.fragments
                    if fragment.is_an_individual
                    and fragment.assigned_identities[0] is None
                ),
                key=lambda x: x.certainty_P2,
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
                for image, episode in zip(fragment.images, fragment.episodes):
                    identities[episode][image] = fragment.identity

        for path, identities_in_episode in zip(self.id_images_file_paths, identities):
            with h5py.File(path, "r+") as file:
                dataset = file.require_dataset(
                    "identities", shape=len(identities_in_episode), dtype=int
                )
                dataset[:] = identities_in_episode

    def get_ordered_list_of_fragments(
        self, scope: str, first_frame_first_global_fragment: int
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
            fragments_subset = [
                fragment
                for fragment in self.fragments
                if fragment.end_frame <= first_frame_first_global_fragment
            ]
            fragments_subset.sort(key=lambda x: x.end_frame, reverse=True)
        elif scope == "to_the_future":
            fragments_subset = [
                fragment
                for fragment in self.fragments
                if fragment.start_frame >= first_frame_first_global_fragment
            ]
            fragments_subset.sort(key=lambda x: x.start_frame, reverse=False)
        else:
            raise ValueError(scope)
        return fragments_subset

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

        # Avoid recursion when saving object on disk
        for fragment in self.fragments:
            fragment.coexisting_individual_fragments.clear()

        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.connect_coexisting_fragments()

    @staticmethod
    def load(path: Path | str) -> "ListOfFragments":
        """Loads a previously saved (see :meth:`save`) from the path
        `path_to_load`
        """
        path = resolve_path(path)
        logging.info(f"Loading ListOfFragments from {path}")
        with open(path, "rb") as file:
            list_of_fragments: "ListOfFragments" = pickle.load(file)

        list_of_fragments.connect_coexisting_fragments()

        return list_of_fragments

    def connect_coexisting_fragments(self):
        logging.info("Connecting coexisting individual fragments")
        # Make it N (not N²) with, maybe, sets (not lists)
        for fragment in track(
            self.fragments, description="Connecting coexisting fragments"
        ):
            fragment.get_coexisting_individual_fragments_indices(self.fragments)

    def get_new_images_and_labels_for_training(self):
        """Extract images and creates labels from every individual fragment
        that has not been used to train the network during the fingerprint
        protocols cascade.

        Returns
        -------
        list
            List of numpy arrays with shape [width, height, num channels]
        list
            labels
        """
        images = []
        labels = []
        for fragment in self.fragments:
            if fragment.acceptable_for_training and not fragment.used_for_training:
                assert fragment.is_an_individual
                images.extend(list(zip(fragment.images, fragment.episodes)))
                labels.extend([fragment.temporary_id] * fragment.number_of_images)
        if len(images) != 0:
            return np.asarray(images), np.asarray(labels)
        return None, None

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
        self.accumulable_individual_fragments = set(
            identifier
            for glob_frag in accumulable_global_fragments
            for identifier in glob_frag.individual_fragments_identifiers
        )
        self.not_accumulable_individual_fragments = (
            set(
                identifier
                for glob_frag in non_accumulable_global_fragments
                for identifier in glob_frag.individual_fragments_identifiers
            )
            - self.accumulable_individual_fragments
        )

        for fragment in self.fragments:
            if fragment.identifier in self.accumulable_individual_fragments:
                fragment.accumulable = True
            elif fragment.identifier in self.not_accumulable_individual_fragments:
                fragment.accumulable = False
            else:
                fragment.accumulable = None

    @property
    def number_of_crossing_fragments(self) -> int:
        return sum(fragment.is_a_crossing for fragment in self.fragments)

    @property
    def number_of_individual_fragments(self) -> int:
        return sum(fragment.is_an_individual for fragment in self.fragments)

    @property
    def number_of_individual_fragments_not_in_a_glob_fragment(self) -> int:
        return sum(
            not fragment.is_in_a_global_fragment and fragment.is_an_individual
            for fragment in self.fragments
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
        return sum(
            fragment.is_an_individual * fragment.number_of_images
            for fragment in self.fragments
        )

    @property
    def number_of_individual_blobs_not_in_a_global_fragment(self) -> int:
        return sum(
            (not fragment.is_in_a_global_fragment and fragment.is_an_individual)
            * fragment.number_of_images
            for fragment in self.fragments
        )

    @property
    def fragments_not_accumulated(self) -> set[int]:
        return self.accumulable_individual_fragments & set(
            fragment.identifier
            for fragment in self.fragments
            if not fragment.used_for_training
        )

    @property
    def number_of_not_accumulated_individual_fragments(self) -> int:
        return len(self.fragments_not_accumulated)

    @property
    def number_of_globally_accumulated_individual_fragments(self) -> int:
        return sum(
            fragment.accumulated_globally and fragment.is_an_individual
            for fragment in self.fragments
        )

    @property
    def number_of_partially_accumulated_individual_fragments(self) -> int:
        return sum(
            fragment.accumulated_partially and fragment.is_an_individual
            for fragment in self.fragments
        )

    @property
    def number_of_accumulable_individual_blobs(self) -> int:
        return sum(
            fragment.accumulable * fragment.number_of_images
            for fragment in self.fragments
            if fragment.accumulable is not None
        )

    @property
    def number_of_not_accumulable_individual_blobs(self) -> int:
        return sum(
            (not fragment.accumulable) * fragment.number_of_images
            for fragment in self.fragments
            if fragment.accumulable is not None
        )

    @property
    def number_of_not_accumulated_individual_blobs(self) -> int:
        return sum(
            fragment.number_of_images
            for fragment in self.fragments
            if fragment.identifier in self.fragments_not_accumulated
        )

    @property
    def number_of_globally_accumulated_individual_blobs(self) -> int:
        return sum(
            (fragment.accumulated_globally and fragment.is_an_individual)
            * fragment.number_of_images
            for fragment in self.fragments
            if fragment.accumulated_globally is not None
        )

    @property
    def number_of_partially_accumulated_individual_blobs(self) -> int:
        return sum(
            (fragment.accumulated_partially and fragment.is_an_individual)
            * fragment.number_of_images
            for fragment in self.fragments
            if fragment.accumulated_partially is not None
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
            "individual_fragments_not_in_a_global_fragment": self.number_of_individual_fragments_not_in_a_glob_fragment,
            "accumulable_individual_fragments": self.number_of_accumulable_individual_fragments,
            "not_accumulable_individual_fragments": self.number_of_not_accumulable_individual_fragments,
            "globally_accumulated_individual_fragments": self.number_of_globally_accumulated_individual_fragments,
            "partially_accumulated_individual_fragments": self.number_of_partially_accumulated_individual_fragments,
            "blobs": self.number_of_blobs,
            "crossing_blobs": self.number_of_crossing_blobs,
            "individual_blobs": self.number_of_individual_blobs,
            "individual_blobs_not_in_a_global_fragment": self.number_of_individual_blobs_not_in_a_global_fragment,
            "accumulable_individual_blobs": self.number_of_accumulable_individual_blobs,
            "not_accumulable_individual_blobs": self.number_of_not_accumulable_individual_blobs,
            "globally_accumulated_individual_blobs": self.number_of_globally_accumulated_individual_blobs,
            "partially_accumulated_individual_blobs": self.number_of_partially_accumulated_individual_blobs,
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
                    len(current.next) > 0
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
                    blob.blob_index,
                    images,
                    centroids,
                    episodes,
                    blob.is_an_individual,
                    number_of_animals,
                )
                used_fragment_identifiers.add(current_fragment_identifier)
                fragments.append(fragment)
        return cls(fragments, id_images_file_paths)

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
            blob.used_for_training = fragment.used_for_training
            blob.accumulation_step = fragment.accumulation_step
            blob.identity_corrected_solving_jumps = (
                fragment.identity_corrected_solving_jumps
            )
            blob.P2_vector = fragment.P2_vector
            blob.user_generated_identity = fragment.user_generated_identity
            blob.is_an_individual = fragment.is_an_individual
