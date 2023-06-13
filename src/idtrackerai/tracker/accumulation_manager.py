import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
from torch.nn import Module

from idtrackerai import Fragment, GlobalFragment, ListOfFragments, ListOfGlobalFragments
from idtrackerai.network import NetworkParams
from idtrackerai.utils import conf, load_id_images

from .identity_network import get_predictions_identities


class AccumulationManager:
    """Manages the process of accumulating images for training the network.

    Attributes
    ----------

    list_of_global_fragments: ListOfGlobalFragments
        Collection of global fragments
    counter : int
        Number of iterations for an instantiation
    certainty_threshold: float
        Value in [0,1] to establish if the identitification of a fragment
        is certain.
    threshold_acceptable_accumulation: float
        Value in [0,1] to establish if an accumulation is acceptable
    accumulation_strategy: string
        Accepts "global" and "partial" in order to perform either partial or
        global accumulation.
    used_images : nd.array
        images used for training the network
    used_labels : nd.array
        labels for the images used for training
    new_images : nd.array
        set of images that will be added to the new training
    new_labels : nd.array
        labels for the set of images that will be added for training
    """

    def __init__(
        self,
        id_images_file_paths: list[Path],
        number_of_animals: int,
        list_of_fragments: ListOfFragments,
        list_of_global_fragments: ListOfGlobalFragments,
        certainty_threshold: float | None = None,
        threshold_acceptable_accumulation: float | None = None,
    ):
        logging.info("Initializing accumulation manager")
        if certainty_threshold is None:
            certainty_threshold = float(conf.CERTAINTY_THRESHOLD)
        if threshold_acceptable_accumulation is None:
            threshold_acceptable_accumulation = float(
                conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            )
        self.id_images_file_paths = id_images_file_paths
        self.number_of_animals = number_of_animals
        self.list_of_fragments = list_of_fragments
        self.list_of_global_fragments = list_of_global_fragments
        self.current_step: int = 0
        self.certainty_threshold = certainty_threshold
        self.threshold_acceptable_accumulation = threshold_acceptable_accumulation
        self.accumulation_strategy: Literal["global", "partial"] = "global"
        self.individual_fragments_used: set[int] = set()
        """set with the individual_fragments_identifiers of the individual
        fragments used for training"""
        self.temporary_individual_fragments_used: set[int] = set()

        self.used_images = None
        self.used_labels = None
        self.new_images = None
        self.new_labels = None
        self.ratio_accumulated_images: float
        # When we init the Accumulation manager we are starting Protocol 1
        # or the accumulation parachute (
        self.threshold_early_stop_accumulation: float = (
            conf.THRESHOLD_EARLY_STOP_ACCUMULATION
        )

    @property
    def new_global_fragments_for_training(self) -> bool:
        """We stop the accumulation when there are not more global fragments
        that are acceptable for training."""
        there_are = any(
            (
                global_fragment.acceptable_for_training(self.accumulation_strategy)
                and not global_fragment.used_for_training
            )
            for global_fragment in self.list_of_global_fragments.global_fragments
        )

        logging.info(
            (
                "[bold]There are global fragments acceptable for training"
                if there_are
                else "[bold]There are no more global fragments acceptable for training"
            ),
            extra={"markup": True},
        )

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
                labels += [fragment.temporary_id] * fragment.number_of_images

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

        ratio = n_images / self.list_of_fragments.number_of_images_in_global_fragments
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
        for i in range(self.number_of_animals):
            if self.new_labels is None:
                new_images_indices = np.asarray([], int)
            else:
                new_images_indices = np.argwhere(self.new_labels == i)[:, 0]

            if self.used_labels is None:
                used_images_indices = np.asarray([], int)
            else:
                used_images_indices = np.argwhere(self.used_labels == i)[:, 0]
            number_of_new_images = len(new_images_indices)
            number_of_used_images = len(used_images_indices)
            number_of_images_for_individual = (
                number_of_new_images + number_of_used_images
            )
            if number_of_images_for_individual > conf.MAXIMAL_IMAGES_PER_ANIMAL:
                # we take a proportion of the old images a new images only if the
                # total number of images for this label is bigger than the
                # limit conf.MAXIMAL_IMAGES_PER_ANIMAL
                number_samples_new = int(
                    conf.MAXIMAL_IMAGES_PER_ANIMAL * conf.RATIO_NEW
                )
                number_samples_used = int(
                    conf.MAXIMAL_IMAGES_PER_ANIMAL * conf.RATIO_OLD
                )
                if number_of_used_images < number_samples_used:
                    # if the proportion of used images is bigger than the number of
                    # used images we take all the used images for this label and update
                    # the number of new images to reach the conf.MAXIMAL_IMAGES_PER_ANIMAL
                    number_samples_used = number_of_used_images
                    number_samples_new = (
                        conf.MAXIMAL_IMAGES_PER_ANIMAL - number_samples_used
                    )
                if number_of_new_images < number_samples_new:
                    # if the proportion of new images is bigger than the number of
                    # new images we take all the new images for this label and update
                    # the number of used images to reac the conf.MAXIMAL_IMAGES_PER_ANIMAL
                    number_samples_new = number_of_new_images
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
                    labels += [i] * number_of_new_images
                if self.used_images is not None:
                    # this condition is set because the first time we accumulate
                    # the variable used_images is None
                    images += list(self.used_images[used_images_indices])
                    labels += [i] * number_of_used_images
        return load_id_images(self.id_images_file_paths, images), np.asarray(labels)

    def update_used_images_and_labels(self):
        """Sets as used the images already used for training"""
        logging.info("Update images and labels used for training")
        if self.current_step == 0:
            self.used_images = self.new_images
            self.used_labels = self.new_labels
        elif self.new_images is not None:
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
        for fragment in self.list_of_fragments.fragments:
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
        for fragment in self.list_of_fragments.fragments:
            if fragment.used_for_training:
                assert fragment.temporary_id is not None
                fragment.identity = fragment.temporary_id + 1
                fragment.set_P1_vector_accumulated()

    def update_set_of_individual_fragments_used(self):
        """Updates the list of individual fragments used for training and
        their identities.
        If an individual fragment was added before is not added again.
        """
        logging.info("Updating list of individual fragments used for training")
        self.individual_fragments_used = {
            fragment.identifier
            for fragment in self.list_of_fragments.fragments
            if fragment.used_for_training
        }

    def split_predictions_after_network_assignment(
        self,
        predictions,
        softmax_probs,
        indices_to_split,
        candidate_individual_fragments_identifiers: list[int],
    ):
        """Gathers predictions relative to fragment images from the GPU and
        splits them according to their organization in fragments.
        """
        logging.debug("Computing fragment prediction statistics")
        individual_fragments_predictions = np.split(predictions, indices_to_split)
        individual_fragments_softmax_probs = np.split(softmax_probs, indices_to_split)

        for (
            individual_fragment_predictions,
            individual_fragment_softmax_probs,
            candidate_individual_fragment_identifier,
        ) in zip(
            individual_fragments_predictions,
            individual_fragments_softmax_probs,
            candidate_individual_fragments_identifiers,
        ):
            self.list_of_fragments.fragments[
                candidate_individual_fragment_identifier
            ].compute_identification_statistics(
                individual_fragment_predictions,
                individual_fragment_softmax_probs,
                self.list_of_fragments.number_of_animals,
            )

    def reset_accumulation_variables(self):
        """After an accumulation is finished reinitialise the variables involved
        in the process.
        """
        self.temporary_individual_fragments_used.clear()
        if self.accumulation_strategy == "global":
            self.number_of_noncertain_global_fragments = 0
            self.number_of_random_assigned_global_fragments = 0
            self.number_of_nonconsistent_global_fragments = 0
            self.number_of_nonunique_global_fragments = 0
        self.number_of_sparse_fragments = 0
        self.number_of_noncertain_fragments = 0
        self.number_of_random_assigned_fragments = 0
        self.number_of_nonconsistent_fragments = 0
        self.number_of_nonunique_fragments = 0
        self.number_of_acceptable_fragments = 0

    def print_accumulation_variables(self):
        lines = (
            "Prediction results:",
            (
                "Non certain global fragments:"
                f" {self.number_of_noncertain_global_fragments}"
            ),
            (
                "Randomly assigned global fragments:"
                f" {self.number_of_random_assigned_global_fragments}"
            ),
            (
                "Non consistent global fragments:"
                f" {self.number_of_nonconsistent_global_fragments}"
            ),
            f"Non unique global fragments: {self.number_of_nonunique_global_fragments}",
            (
                "Acceptable global fragments:"
                f" {self.number_of_acceptable_global_fragments}"
            ),
            f"Non certain fragments: {self.number_of_noncertain_fragments}",
            f"Randomly assigned fragments: {self.number_of_random_assigned_fragments}",
            f"Non consistent fragments: {self.number_of_nonconsistent_fragments}",
            f"Non unique fragments: {self.number_of_nonunique_fragments}",
            f"Acceptable fragments: {self.number_of_acceptable_fragments}",
        )
        logging.info("\n    ".join(lines))

    def get_acceptable_global_fragments_for_training(
        self,
        candidate_individual_fragments_identifiers: list[int],
        accumulation_trial: int,
    ):
        """Assigns identities during test to individual fragments and rank them
        according to the score computed from the certainty of identification
        and the minimum distance traveled.

        Parameters
        ----------
        candidate_individual_fragments_identifiers : list
            List of fragment identifiers.
        """
        self.accumulation_strategy = "global"
        self.candidate_individual_fragments_identifiers = (
            candidate_individual_fragments_identifiers
        )
        self.reset_accumulation_variables()
        logging.debug("Accumulating by global strategy")
        for global_fragment in self.list_of_global_fragments.global_fragments:
            if not global_fragment.used_for_training:
                self.check_if_is_globally_acceptable_for_training(global_fragment)

        self.number_of_acceptable_global_fragments = sum(
            global_fragment.acceptable_for_training(self.accumulation_strategy)
            and not global_fragment.used_for_training
            for global_fragment in self.list_of_global_fragments.global_fragments
        )
        if accumulation_trial == 0:
            min_number_of_imgs_accumulated_to_start_partial_accumulation = (
                conf.MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION
            )
        else:
            min_number_of_imgs_accumulated_to_start_partial_accumulation = 0
        if (
            self.number_of_acceptable_global_fragments == 0
            and self.ratio_accumulated_images
            > min_number_of_imgs_accumulated_to_start_partial_accumulation
            and self.ratio_accumulated_images < self.threshold_early_stop_accumulation
        ):
            logging.debug("Accumulating by partial strategy")
            self.accumulation_strategy = "partial"
            self.reset_accumulation_variables()
            for global_fragment in self.list_of_global_fragments.global_fragments:
                if not global_fragment.used_for_training:
                    self.check_if_is_partially_acceptable_for_training(global_fragment)
        elif (
            self.ratio_accumulated_images
            < min_number_of_imgs_accumulated_to_start_partial_accumulation
        ):
            logging.info(
                "The ratio of accumulated images is too small and a partial"
                " accumulation might fail."
            )

    def reset_non_acceptable_fragment(self, fragment: Fragment):
        """Resets the collection of non-acceptable fragments.

        Parameters
        ----------
        fragment : Fragment object
            Collection of images related to the same individual
        """
        if (
            fragment.identifier not in self.temporary_individual_fragments_used
            and fragment.identifier not in self.individual_fragments_used
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
        for fragment in global_fragment.individual_fragments:
            self.reset_non_acceptable_fragment(fragment)

    def check_if_is_globally_acceptable_for_training(
        self, global_fragment: GlobalFragment
    ):
        assert self.accumulation_strategy == "global"

        for fragment in global_fragment.individual_fragments:
            fragment.acceptable_for_training = True

        for fragment in global_fragment.individual_fragments:
            if fragment.identifier in self.candidate_individual_fragments_identifiers:
                if fragment.certainty < self.certainty_threshold:
                    # if the certainty of the individual fragment is not high enough
                    # we set the global fragment to be non-acceptable for training
                    self.reset_non_acceptable_global_fragment(global_fragment)
                    self.number_of_noncertain_global_fragments += 1
                    fragment.is_certain = False
                    break
                # if the certainty of the individual fragment is high enough
                fragment.is_certain = True
            elif fragment.identifier in self.individual_fragments_used:
                # if the individual fragment is not in the list of
                # candidates is because it has been assigned
                # and it is in the list of individual_fragments_used.
                # We set the certainty to 1. And we
                fragment.is_certain = True
            else:
                logging.warning(
                    "Individual fragment not in candidates or in used, this should"
                    " not happen"
                )
        # Compute identities if the global_fragment is certain
        if not global_fragment.acceptable_for_training("global"):
            return

        P1_array, index_individual_fragments_sorted_by_P1_max_to_min = (
            get_P1_array_and_argsort(global_fragment)
        )
        # set to zero the P1 of the the identities of the individual
        # fragments that have been already used
        for index_individual_fragment, fragment in enumerate(
            global_fragment.individual_fragments
        ):
            if (
                fragment.identifier in self.individual_fragments_used
                or fragment.identifier in self.temporary_individual_fragments_used
            ):
                P1_array[index_individual_fragment, :] = 0.0
                P1_array[:, fragment.temporary_id] = 0.0
        # assign temporal identity to individual fragments by hierarchical P1
        for (
            index_individual_fragment
        ) in index_individual_fragments_sorted_by_P1_max_to_min:
            fragment = global_fragment.individual_fragments[index_individual_fragment]
            assert isinstance(fragment, Fragment)
            if fragment.temporary_id is None:
                if p1_below_random(P1_array, index_individual_fragment, fragment):
                    fragment.P1_below_random = True
                    self.number_of_random_assigned_global_fragments += 1
                    self.reset_non_acceptable_global_fragment(global_fragment)
                    break

                temporary_id = np.argmax(P1_array[index_individual_fragment, :])
                if not fragment.check_consistency_with_coexistent_individual_fragments(
                    temporary_id
                ):
                    self.reset_non_acceptable_global_fragment(global_fragment)
                    fragment.non_consistent = True
                    self.number_of_nonconsistent_global_fragments += 1
                    break

                P1_array = set_fragment_temporary_id(
                    fragment, int(temporary_id), P1_array, index_individual_fragment
                )

        # Check if the global fragment is unique after assigning the identities
        if global_fragment.acceptable_for_training("global"):
            if not global_fragment.is_unique(self.number_of_animals):
                # set acceptable_for_training to False and temporary_id to
                # None for all the individual_fragments
                # that had not been accumulated before (i.e. not in
                # temporary_individual_fragments_used or individual_fragments_used)
                self.reset_non_acceptable_global_fragment(global_fragment)
                self.number_of_nonunique_global_fragments += 1
            else:
                global_fragment.accumulation_step = self.current_step
                self.temporary_individual_fragments_used.update(
                    fragment.identifier
                    for fragment in global_fragment.individual_fragments
                    if fragment.identifier not in self.individual_fragments_used
                )

    def check_if_is_partially_acceptable_for_training(
        self, global_fragment: GlobalFragment
    ):
        assert self.accumulation_strategy == "partial"
        for fragment in global_fragment.individual_fragments:
            fragment.acceptable_for_training = False

        for fragment in global_fragment.individual_fragments:
            # Check certainties of the individual fragme
            if fragment.identifier in self.candidate_individual_fragments_identifiers:
                if fragment.has_enough_accumulated_coexisting_fragments:
                    # Check if the more than half of the individual fragments
                    # that coexist with this one have being accumulated
                    if fragment.certainty < self.certainty_threshold:
                        # if the certainty of the individual fragment is not high enough
                        # we set the global fragment not to be acceptable for training
                        self.reset_non_acceptable_fragment(fragment)
                        self.number_of_noncertain_fragments += 1
                        fragment.is_certain = False
                    else:
                        # if the certainty of the individual fragment is high enough
                        fragment.is_certain = True
                        fragment.acceptable_for_training = True
                else:
                    self.reset_non_acceptable_fragment(fragment)
                    self.number_of_sparse_fragments += 1
            elif fragment.identifier in self.individual_fragments_used:
                # if the individual fragment is not in the list of candidates
                # is because it has been assigned
                # and it is in the list of individual_fragments_used.
                # We set the certainty to 1. And we
                fragment.is_certain = True
            else:
                logging.warning(
                    "Individual fragment not in candidates or in used, this should"
                    " not happen"
                )

        # Compute identities if the global_fragment is certain
        # get array of P1 values for the global fragment
        P1_array = np.asarray(
            [fragment.P1_vector for fragment in global_fragment.individual_fragments]
        )
        # get the maximum P1 of each individual fragment
        P1_max = np.max(P1_array, axis=1)
        # logging.debug("P1 max: %s" %str(P1_max))
        # get the index position of the individual fragments ordered by
        # P1_max from max to min
        index_individual_fragments_sorted_by_P1_max_to_min = np.argsort(P1_max)[::-1]
        # set to zero the P1 of the the identities of the individual
        # fragments that have been already used
        for index_individual_fragment, fragment in enumerate(
            global_fragment.individual_fragments
        ):
            if (
                fragment.identifier in self.individual_fragments_used
                or fragment.identifier in self.temporary_individual_fragments_used
            ):
                P1_array[index_individual_fragment, :] = 0.0
                P1_array[:, fragment.temporary_id] = 0.0

        # assign temporary identity to individual fragments by hierarchical P1
        for (
            index_individual_fragment
        ) in index_individual_fragments_sorted_by_P1_max_to_min:
            fragment = global_fragment.individual_fragments[index_individual_fragment]
            assert isinstance(fragment, Fragment)  # for PyLance

            if fragment.temporary_id is None and fragment.acceptable_for_training:
                if (
                    np.max(P1_array[index_individual_fragment, :])
                    < 1.0 / fragment.number_of_images
                ):
                    fragment.P1_below_random = True
                    self.number_of_random_assigned_fragments += 1
                    self.reset_non_acceptable_fragment(fragment)
                else:
                    temporary_id = np.argmax(P1_array[index_individual_fragment, :])
                    if not fragment.check_consistency_with_coexistent_individual_fragments(
                        temporary_id
                    ):
                        self.reset_non_acceptable_fragment(fragment)
                        fragment.non_consistent = True
                        self.number_of_nonconsistent_fragments += 1
                    else:
                        fragment.acceptable_for_training = True
                        fragment.temporary_id = int(temporary_id)
                        P1_array[index_individual_fragment, :] = 0.0
                        P1_array[:, temporary_id] = 0.0

        # Check if the global fragment is unique after assigning the identities
        if not global_fragment.is_partially_unique:
            for fragment in global_fragment.individual_fragments:
                if fragment.temporary_id in global_fragment.duplicated_identities:
                    self.reset_non_acceptable_fragment(fragment)
                    self.number_of_nonunique_fragments += 1

        self.temporary_individual_fragments_used.update(
            fragment.identifier
            for fragment in global_fragment.individual_fragments
            if fragment.acceptable_for_training
            and fragment.identifier not in self.individual_fragments_used
        )
        self.number_of_acceptable_fragments += sum(
            bool(fragment.acceptable_for_training) and not fragment.used_for_training
            for fragment in global_fragment.individual_fragments
        )
        global_fragment.accumulation_step = self.current_step
        assert all(
            fragment.temporary_id is not None
            for fragment in global_fragment.individual_fragments
            if fragment.acceptable_for_training and fragment.is_an_individual
        )


def get_predictions_of_candidates_fragments(
    identification_model: Module,
    id_images_file_paths: list[Path],
    network_params: NetworkParams,
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
    candidate_individual_fragments_identifiers: list[int] = []

    for fragment in list_of_fragments.individual_fragments:
        if not fragment.used_for_training:
            images += fragment.image_locations
            lengths.append(fragment.number_of_images)
            candidate_individual_fragments_identifiers.append(fragment.identifier)

    assert images
    images = load_id_images(id_images_file_paths, images)

    predictions, softmax_probs = get_predictions_identities(
        identification_model, images, network_params
    )

    assert sum(lengths) == len(predictions)
    return (
        predictions,
        softmax_probs,
        np.cumsum(lengths)[:-1],
        candidate_individual_fragments_identifiers,
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
    index_individual_fragments_sorted_by_P1_max_to_min : nd.array
        Argsort of P1 array of each individual fragment
    """
    # get array of P1 values for the global fragment
    P1_array = np.asarray(
        [fragment.P1_vector for fragment in global_fragment.individual_fragments]
    )
    # get the maximum P1 of each individual fragment
    P1_max = np.max(P1_array, axis=1)
    # logging.debug("P1 max: %s" %str(P1_max))
    # get the index position of the individual fragments ordered by P1_max
    # from max to min
    index_individual_fragments_sorted_by_P1_max_to_min = np.argsort(P1_max)[::-1]
    return P1_array, index_individual_fragments_sorted_by_P1_max_to_min


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
    return (
        np.max(P1_array[index_individual_fragment, :]) < 1.0 / fragment.number_of_images
    )


def set_fragment_temporary_id(
    fragment: Fragment,
    temporary_id: int,
    P1_array: np.ndarray,
    index_individual_fragment: int,
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
    fragment.temporary_id = int(temporary_id)
    P1_array[index_individual_fragment, :] = 0.0
    P1_array[:, temporary_id] = 0.0
    return P1_array
