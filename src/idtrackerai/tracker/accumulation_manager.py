import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
from torch.nn import Module

from .. import Fragment, GlobalFragment, ListOfFragments, ListOfGlobalFragments
from ..utils import conf, load_id_images
from .identity_network import get_predictions_identities

AccStrategy = Literal["global", "partial"]


class AccumulationManager:
    "Manages the process of accumulating images for training the network"

    accumulation_strategy: AccStrategy
    id_images_file_paths: list[Path]
    n_animals: int
    list_of_fragments: ListOfFragments
    list_of_global_fragments: ListOfGlobalFragments
    current_step: int
    accumulation_strategy: AccStrategy = "global"
    temporary_used_fragments: set[int]

    used_images: np.ndarray | None = None
    used_labels: np.ndarray | None = None
    new_images: np.ndarray | None = None
    new_labels: np.ndarray | None = None
    ratio_accumulated_images: float

    n_noncertain_global_fragments: int
    n_random_assigned_global_fragments: int
    n_nonconsistent_global_fragments: int
    n_nonunique_global_fragments: int
    n_sparse_fragments: int
    n_noncertain_fragments: int
    n_random_assigned_fragments: int
    n_nonconsistent_fragments: int
    n_nonunique_fragments: int
    n_acceptable_fragments: int

    def __init__(
        self,
        id_images_file_paths: list[Path],
        n_animals: int,
        list_of_fragments: ListOfFragments,
        list_of_global_fragments: ListOfGlobalFragments,
    ):
        logging.info("Initializing accumulation manager")

        self.id_images_file_paths = id_images_file_paths
        self.n_animals = n_animals
        self.list_of_fragments = list_of_fragments
        self.list_of_global_fragments = list_of_global_fragments

        self.current_step = 0

    @property
    def new_global_fragments_for_training(self) -> bool:
        """We stop the accumulation when there are not more global fragments
        that are acceptable for training."""
        there_are = any(
            (
                global_fragment.acceptable_for_training(self.accumulation_strategy)
                and not global_fragment.used_for_training
            )
            for global_fragment in self.list_of_global_fragments
        )

        if there_are:
            log = "There are global fragments acceptable for training"
        else:
            log = "There are no more global fragments acceptable for training"
        if self.accumulation_strategy == "partial":
            log = log.replace("global fragments", "fragments")
        logging.info("[bold]" + log, extra={"markup": True})

        return there_are

    def get_new_images_and_labels(self):
        """Get the images and labels of the new global fragments that are going
        to be used for training. This function checks whether the images of a individual
        fragment have been added before"""

        images = []
        labels = []
        for fragment in self.list_of_fragments.individual_fragments:
            if fragment.acceptable_for_training and not fragment.used_for_training:
                images += fragment.image_locations
                labels += [fragment.temporary_id] * fragment.n_images

        if images:
            self.new_images, self.new_labels = np.asarray(images), np.asarray(labels)
        else:
            self.new_images, self.new_labels = None, None

        n_used_images = len(self.used_images) if self.used_images is not None else 0
        n_new_images = len(self.new_images) if self.new_images is not None else 0
        n_images = n_used_images + n_new_images

        if n_new_images:
            logging.info("%d new images for training", n_new_images)
        else:
            logging.info("There are no new images in this accumulation")

        if n_used_images:
            logging.info("%d old images for training", n_used_images)

        ratio = n_images / self.list_of_fragments.n_images_in_global_fragments
        logging.info(
            f"{n_images} images in total, {ratio:.2%} of the total accumulable"
        )

    def get_images_and_labels_for_training(self):
        """Create a new dataset of labelled images to train the idCNN in the
        following way:
        Per individual select conf.MAXIMAL_IMAGES_PER_ANIMAL images.
        Such collection of images is composed
        of a ratio corresponding to conf.RATIO_NEW of new images (acquired in
        the current evaluation of the
        global fragments) and conf.RATIO_OLD of images already used
        in the previous iteration."""
        logging.info("Getting images for training...")
        random.seed(0)
        images = []
        labels = []
        for i in range(self.n_animals):
            if self.new_labels is None:
                new_images_indices = np.empty(0, int)
                # avoid default int32 type in some computers
            else:
                new_images_indices = np.nonzero(self.new_labels == i)[0]

            if self.used_labels is None:
                used_images_indices = np.empty(0, int)
            else:
                used_images_indices = np.nonzero(self.used_labels == i)[0]
            n_new_images = len(new_images_indices)
            n_used_images = len(used_images_indices)
            n_images_for_individual = n_new_images + n_used_images
            if n_images_for_individual > conf.MAXIMAL_IMAGES_PER_ANIMAL:
                # we take a proportion of the old images a new images only if the
                # total number of images for this label is bigger than the
                # limit conf.MAXIMAL_IMAGES_PER_ANIMAL
                number_samples_new = int(
                    conf.MAXIMAL_IMAGES_PER_ANIMAL * conf.RATIO_NEW
                )
                number_samples_used = (
                    conf.MAXIMAL_IMAGES_PER_ANIMAL - number_samples_new
                )
                if n_used_images < number_samples_used:
                    # if the proportion of used images is bigger than the number of
                    # used images we take all the used images for this label and update
                    # the number of new images to reach the conf.MAXIMAL_IMAGES_PER_ANIMAL
                    number_samples_used = n_used_images
                    number_samples_new = (
                        conf.MAXIMAL_IMAGES_PER_ANIMAL - number_samples_used
                    )
                if n_new_images < number_samples_new:
                    # if the proportion of new images is bigger than the number of
                    # new images we take all the new images for this label and update
                    # the number of used images to reac the conf.MAXIMAL_IMAGES_PER_ANIMAL
                    number_samples_new = n_new_images
                    number_samples_used = (
                        conf.MAXIMAL_IMAGES_PER_ANIMAL - number_samples_new
                    )
                # we put together a random sample of the new images and the used images
                if self.new_images is not None:
                    images += random.sample(
                        list(self.new_images[new_images_indices]), number_samples_new
                    )
                    labels += [i] * number_samples_new
                if self.used_images is not None:
                    # this condition is set because the first time we accumulate
                    # the variable used_images is None
                    images += random.sample(
                        list(self.used_images[used_images_indices]), number_samples_used
                    )
                    labels += [i] * number_samples_used
            else:
                # if the total number of images for this label does not exceed
                # the conf.MAXIMAL_IMAGES_PER_ANIMAL
                # we take all the new images and all the used images
                if self.new_images is not None:
                    images += list(self.new_images[new_images_indices])
                    labels += [i] * n_new_images
                if self.used_images is not None:
                    # this condition is set because the first time we accumulate
                    # the variable used_images is None
                    images += list(self.used_images[used_images_indices])
                    labels += [i] * n_used_images
        return load_id_images(self.id_images_file_paths, images), np.asarray(
            labels, dtype=np.int64
        )

    def update_used_images_and_labels(self):
        """Sets as used the images already used for training"""
        logging.info("Update images and labels used for training")
        if self.current_step == 0:
            self.used_images = self.new_images
            self.used_labels = self.new_labels
        elif self.new_images is not None:
            assert self.used_images is not None
            assert self.used_labels is not None
            assert self.new_labels is not None
            self.used_images = np.concatenate(
                (self.used_images, self.new_images), axis=0
            )
            self.used_labels = np.concatenate(
                [self.used_labels, self.new_labels], axis=0
            )

    def update_fragments_used_for_training(self):
        """Once a global fragment has been used for training, sets the flags
        used_for_training to TRUE and acceptable_for_training to FALSE"""
        logging.info("Updating fragments used for training")
        for fragment in self.list_of_fragments:
            if fragment.acceptable_for_training and not fragment.used_for_training:
                fragment.used_for_training = True
                fragment.acceptable_for_training = False
                fragment.set_partially_or_globally_accumulated(
                    self.accumulation_strategy
                )
                fragment.accumulation_step = self.current_step

    def assign_identities_to_fragments_used_for_training(self):
        """Assign the identities to the global fragments used for training and
        their individual fragments.
        This function checks that the identities of the individual fragments in
        the global fragment
        are consistent with the previously assigned identities
        """
        logging.info("Assigning identities to accumulated global fragments")
        for fragment in self.list_of_fragments:
            if fragment.used_for_training:
                assert fragment.temporary_id is not None
                fragment.identity = fragment.temporary_id + 1
                fragment.P1_vector[:] = 0.0
                fragment.P1_vector[fragment.temporary_id] = 1.0

    def split_predictions_after_network_assignment(
        self,
        predictions: np.ndarray,
        softmax_probs: np.ndarray,
        indices_to_split: np.ndarray,
        candidate_fragments_identifiers: list[int],
    ):
        """Gathers predictions relative to fragment images from the GPU and
        splits them according to their organization in fragments.
        """
        logging.debug("Computing fragment prediction statistics")
        fragments_predictions = np.split(predictions, indices_to_split)
        fragments_softmax_probs = np.split(softmax_probs, indices_to_split)

        for predictions, softmax_probs, identifier in zip(
            fragments_predictions,
            fragments_softmax_probs,
            candidate_fragments_identifiers,
        ):
            self.list_of_fragments.fragments[
                identifier
            ].compute_identification_statistics(
                predictions, softmax_probs, self.n_animals
            )

    def reset_accumulation_variables(self):
        """After an accumulation is finished reinitialise the variables involved
        in the process.
        """
        self.temporary_used_fragments = set()
        self.n_noncertain_global_fragments = 0
        self.n_random_assigned_global_fragments = 0
        self.n_nonconsistent_global_fragments = 0
        self.n_nonunique_global_fragments = 0
        self.n_sparse_fragments = 0
        self.n_noncertain_fragments = 0
        self.n_random_assigned_fragments = 0
        self.n_nonconsistent_fragments = 0
        self.n_nonunique_fragments = 0
        self.n_acceptable_fragments = 0

    def log_global_accumulation_variables(self):
        lines = (
            "Global prediction results:",
            f"Acceptable global fragments: {self.n_acceptable_global_fragments}",
            (
                "Non acceptable global fragments:"
                f" {self.n_noncertain_global_fragments+self.n_random_assigned_global_fragments+self.n_nonconsistent_global_fragments+self.n_nonunique_global_fragments}"
            ),
            f"    Non certain: {self.n_noncertain_global_fragments}",
            f"    Non significant: {self.n_random_assigned_global_fragments}",
            f"    Non consistent: {self.n_nonconsistent_global_fragments}",
            f"    Non unique: {self.n_nonunique_global_fragments}",
        )
        logging.info("\n    ".join(lines))

    def log_partial_accumulation_variables(self):
        lines = (
            "Partial prediction results:",
            f"Acceptable fragments: {self.n_acceptable_fragments}",
            (
                "Non acceptable fragments:"
                f" {self.n_noncertain_fragments+self.n_random_assigned_fragments+self.n_nonconsistent_fragments+self.n_nonunique_fragments+self.n_sparse_fragments}"
            ),
            f"    Non certain: {self.n_noncertain_fragments}",
            f"    Non significant: {self.n_random_assigned_fragments}",
            f"    Non consistent: {self.n_nonconsistent_fragments}",
            f"    Non unique: {self.n_nonunique_fragments}",
            f"    Too sparse: {self.n_sparse_fragments}",
        )
        logging.info("\n    ".join(lines))

    def assign_identities(self, accumulation_trial: int):
        """Assigns identities during test to individual fragments and rank them
        according to the score computed from the certainty of identification
        and the minimum distance traveled."""
        self.reset_accumulation_variables()
        self.accumulation_strategy = "global"
        logging.debug(
            "Accumulating by [bold]global[/] strategy", extra={"markup": True}
        )
        for global_fragment in self.list_of_global_fragments:
            self.check_if_is_globally_acceptable_for_training(global_fragment)

        self.n_acceptable_global_fragments = sum(
            global_fragment.acceptable_for_training(self.accumulation_strategy)
            and not global_fragment.used_for_training
            for global_fragment in self.list_of_global_fragments
        )

        self.log_global_accumulation_variables()

        if self.n_acceptable_global_fragments > 0:
            logging.info("Global accumulation succeeded")
            return
        logging.info("Global accumulation failed")

        if (
            accumulation_trial == 0
            and self.ratio_accumulated_images
            < conf.MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION
        ):
            logging.info(
                f"The ratio of accumulated images ({self.ratio_accumulated_images:.2%})"
                " is too small and a partial accumulation might fail."
            )
            return

        if self.ratio_accumulated_images > conf.THRESHOLD_EARLY_STOP_ACCUMULATION:
            logging.info(
                f"The ratio of accumulated images ({self.ratio_accumulated_images:.2%})"
                f" is higher than {conf.THRESHOLD_EARLY_STOP_ACCUMULATION:.2%}, early"
                " stopping"
            )
            return

        logging.debug(
            "Accumulating by [bold]partial[/] strategy", extra={"markup": True}
        )
        self.accumulation_strategy = "partial"
        for global_fragment in self.list_of_global_fragments:
            if not global_fragment.used_for_training:
                self.check_if_is_partially_acceptable_for_training(global_fragment)
        self.log_partial_accumulation_variables()

    def reset_non_acceptable_fragment(self, fragment: Fragment):
        """Resets the collection of non-acceptable fragments.

        Parameters
        ----------
        fragment : Fragment object
            Collection of images related to the same individual
        """
        if (
            fragment.identifier not in self.temporary_used_fragments
            and not fragment.used_for_training
        ):
            fragment.temporary_id = None
            fragment.acceptable_for_training = False

    def reset_non_acceptable_global_fragment(self, global_fragment: GlobalFragment):
        """Reset the flag for non-accpetable global fragments.

        Parameters
        ----------
        global_fragment : GlobalFragment object
            Collection of images relative to a part of the video in which all the animals are visible.
        """
        for fragment in global_fragment:
            self.reset_non_acceptable_fragment(fragment)

    def check_if_is_globally_acceptable_for_training(
        self, global_fragment: GlobalFragment
    ):
        if global_fragment.used_for_training:
            return
        assert self.accumulation_strategy == "global"

        for fragment in global_fragment:
            fragment.acceptable_for_training = True

        for fragment in global_fragment:
            if fragment.used_for_training:
                continue
            if not fragment.is_certain:
                # if the certainty of the individual fragment is not high enough
                # we set the global fragment to be non-acceptable for training
                self.reset_non_acceptable_global_fragment(global_fragment)
                self.n_noncertain_global_fragments += 1
                break

        # Compute identities if the global_fragment is certain
        if not global_fragment.acceptable_for_training("global"):
            return

        P1_array, indices_sorted_by_P1 = get_P1_array_and_argsort(global_fragment)
        # set to zero the P1 of the the identities of the individual
        # fragments that have been already used
        for fragment_index, fragment in enumerate(global_fragment):
            if (
                fragment.used_for_training
                or fragment.identifier in self.temporary_used_fragments
            ):
                P1_array[fragment_index] = 0.0
                P1_array[:, fragment.temporary_id] = 0.0
            # TODO exclusive ROIs could go here?

        # assign temporal identity to individual fragments by hierarchical P1
        for fragment_index in indices_sorted_by_P1:
            fragment: Fragment = global_fragment.fragments[fragment_index]
            if fragment.temporary_id is not None:
                continue
            if p1_below_random(P1_array, fragment_index, fragment):
                fragment.P1_below_random = True
                self.n_random_assigned_global_fragments += 1
                self.reset_non_acceptable_global_fragment(global_fragment)
                break

            temporary_id = np.argmax(P1_array[fragment_index])
            if fragment.is_inconsistent_with_coexistent_fragments(temporary_id):
                self.reset_non_acceptable_global_fragment(global_fragment)
                fragment.non_consistent = True
                self.n_nonconsistent_global_fragments += 1
                break

            P1_array = set_fragment_temporary_id(
                fragment, int(temporary_id), P1_array, fragment_index
            )

        # Check if the global fragment is unique after assigning the identities
        if not global_fragment.acceptable_for_training("global"):
            return

        if global_fragment.is_unique(self.n_animals):
            global_fragment.accumulation_step = self.current_step
            self.temporary_used_fragments.update(
                fragment.identifier
                for fragment in global_fragment
                if not fragment.used_for_training
            )
        else:
            # set acceptable_for_training to False and temporary_id to
            # None for all the individual_fragments
            # that had not been accumulated before (i.e. not in
            # temporary_individual_fragments_used or individual_fragments_used)
            self.reset_non_acceptable_global_fragment(global_fragment)
            self.n_nonunique_global_fragments += 1

    def check_if_is_partially_acceptable_for_training(
        self, global_fragment: GlobalFragment
    ):
        assert self.accumulation_strategy == "partial"
        for fragment in global_fragment:
            fragment.acceptable_for_training = False

        for fragment in global_fragment:
            if fragment.used_for_training:
                continue

            if fragment.has_enough_accumulated_coexisting_fragments:
                if not fragment.is_certain:
                    self.reset_non_acceptable_fragment(fragment)
                    self.n_noncertain_fragments += 1
                else:
                    fragment.acceptable_for_training = True
            else:
                self.reset_non_acceptable_fragment(fragment)
                self.n_sparse_fragments += 1

        # Compute identities if the global_fragment is certain
        # get array of P1 values for the global fragment

        P1_array, indices_sorted_by_P1 = get_P1_array_and_argsort(global_fragment)
        # set to zero the P1 of the the identities of the individual
        # fragments that have been already used
        for fragment_index, fragment in enumerate(global_fragment):
            if (
                fragment.used_for_training
                or fragment.identifier in self.temporary_used_fragments
            ):
                P1_array[fragment_index] = 0.0
                P1_array[:, fragment.temporary_id] = 0.0

        # assign temporary identity to individual fragments by hierarchical P1
        for fragment_index in indices_sorted_by_P1:
            fragment: Fragment = global_fragment.fragments[fragment_index]

            if fragment.temporary_id is None and fragment.acceptable_for_training:
                if p1_below_random(P1_array, fragment_index, fragment):
                    fragment.P1_below_random = True
                    self.n_random_assigned_fragments += 1
                    self.reset_non_acceptable_fragment(fragment)
                else:
                    temporary_id = np.argmax(P1_array[fragment_index])
                    if fragment.is_inconsistent_with_coexistent_fragments(temporary_id):
                        self.reset_non_acceptable_fragment(fragment)
                        fragment.non_consistent = True
                        self.n_nonconsistent_fragments += 1
                    else:
                        fragment.acceptable_for_training = True
                        fragment.temporary_id = int(temporary_id)
                        P1_array[fragment_index] = 0.0
                        P1_array[:, temporary_id] = 0.0

        # Check if the global fragment is unique after assigning the identities
        if not global_fragment.is_partially_unique:
            for fragment in global_fragment:
                if fragment.temporary_id in global_fragment.duplicated_identities:
                    self.reset_non_acceptable_fragment(fragment)
                    self.n_nonunique_fragments += 1

        self.temporary_used_fragments.update(
            fragment.identifier
            for fragment in global_fragment
            if fragment.acceptable_for_training and not fragment.used_for_training
        )
        self.n_acceptable_fragments += sum(
            bool(fragment.acceptable_for_training) and not fragment.used_for_training
            for fragment in global_fragment
        )
        global_fragment.accumulation_step = self.current_step
        assert all(
            fragment.temporary_id is not None
            for fragment in global_fragment
            if fragment.acceptable_for_training and fragment.is_an_individual
        )


def get_predictions_of_candidates_fragments(
    identification_model: Module,
    id_images_file_paths: list[Path],
    list_of_fragments: ListOfFragments,
):
    """Get predictions of individual fragments that have been used to train the
    idCNN in an accumulation's iteration

    Parameters
    ----------
    net : ConvNetwork object
        network used to identify the animals
    video : Video object
        Object containing all the parameters of the video.
    fragments : list
        List of fragment objects

    Returns
    -------
    assigner._predictions  : nd.array
        predictions associated to each image organised by individual fragments
    assigner._softmax_probs : np.array
        softmax vector associated to each image organised by individual fragments
    np.cumsum(lengths)[:-1]  : nd.array
        cumulative sum of the number of images contained in every fragment
        (used to rebuild the collection of images per fragment after gathering
        predicions and softmax vectors from the gpu)
    candidate_individual_fragments_identifiers : list
        list of fragment identifiers
    """
    images = []
    lengths = []
    candidate_fragments_identifiers: list[int] = []

    for fragment in list_of_fragments.individual_fragments:
        if not fragment.used_for_training:
            images += fragment.image_locations
            lengths.append(fragment.n_images)
            candidate_fragments_identifiers.append(fragment.identifier)

    assert images
    images = load_id_images(id_images_file_paths, images)

    predictions, softmax_probs = get_predictions_identities(
        identification_model, images
    )

    assert sum(lengths) == len(predictions)
    return (
        predictions,
        softmax_probs,
        np.cumsum(lengths)[:-1],
        candidate_fragments_identifiers,
    )


def get_P1_array_and_argsort(global_fragment: GlobalFragment):
    """Given a global fragment computes P1 for each of its individual
    fragments and returns a
    matrix of sorted indices according to P1

    Parameters
    ----------
    global_fragment : GlobalFragment object
        Collection of images relative to a part of the video in which all
        the animals are visible.

    Returns
    -------
    P1_array : nd.array
        P1 computed for every individual fragment in the global fragment
    index_individual_fragments_sorted_by_P1 : nd.array
        Argsort of P1 array of each individual fragment
    """
    P1_array = np.array([fragment.P1_vector for fragment in global_fragment])
    return P1_array, np.argsort(P1_array.max(1))[::-1]


def p1_below_random(
    P1_array: np.ndarray, index_individual_fragment: np.ndarray, fragment: Fragment
):
    """Evaluate if a fragment has been assigned with a certainty lower than
    random (wrt the number of possible identities)

    Parameters
    ----------
    P1_array  : nd.array
        P1 vector of a fragment object
    index_individual_fragment  : nd.array
        Argsort of the P1 array of fragment
    fragment : Fragment
        Fragment object containing images associated with a single individual

    Returns
    -------
    p1_below_random_flag : bool
        True if a fragment has been identified with a certainty below random
    """
    return P1_array[index_individual_fragment].max() < (1.0 / fragment.n_images)


def set_fragment_temporary_id(
    fragment: Fragment, temporary_id: int, P1_array: np.ndarray, fragment_index: int
):
    """Given a P1 array relative to a global fragment sets to 0 the row
    relative to fragment
    which is temporarily identified with identity temporary_id

    Parameters
    ----------
    fragment : Fragment
        Fragment object containing images associated with a single individual
    temporary_id : int
        temporary identifier associated to fragment
    P1_array  : nd.array
        P1 vector of fragment
    index_individual_fragment : int
        Index of fragment with respect to a global fragment in which it is
        contained

    Returns
    -------
    P1_array  : nd.array
        updated P1 array
    """
    fragment.temporary_id = temporary_id
    P1_array[fragment_index] = 0.0
    P1_array[:, temporary_id] = 0.0
    return P1_array
