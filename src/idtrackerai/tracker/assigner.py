"""Identification of individual fragments given the predictions generate by the idCNN
"""
import logging

import numpy as np
from torch import nn

from idtrackerai import Fragment, ListOfFragments
from idtrackerai.network import NetworkParams

from .network.get_predictions import get_predictions_identities


def compute_identification_statistics_for_non_accumulated_fragments(
    fragments: list[Fragment],
    all_predictions: np.ndarray,
    all_softmax_probs: np.ndarray,
    number_of_animals=None,
):
    """Given the predictions associated to the images in each (individual)
    fragment in the list fragments if computes the statistics necessary for the
    identification of fragment.

    Parameters
    ----------
    fragments : list
        List of individual fragment objects
    assigner : <GetPrediction object>
        The assigner object has as main attributes the list of predictions
        associated to `images` and the the corresponding softmax vectors
    number_of_animals : int
        number of animals to be tracked
    """
    counter = 0
    for fragment in fragments:
        if not fragment.used_for_training and fragment.is_an_individual:
            next_counter_value = counter + fragment.number_of_images
            predictions = all_predictions[counter:next_counter_value]
            softmax_probs = all_softmax_probs[counter:next_counter_value]
            fragment.compute_identification_statistics(
                predictions, softmax_probs, number_of_animals=number_of_animals
            )
            counter = next_counter_value


def assign_identity(list_of_fragments: ListOfFragments):
    """Identifies the individual fragments recursively, based on the value of
    P2

    Parameters
    ----------
    list_of_fragments : <ListOfFragments object>
        collection of the individual fragments and associated methods
    """
    logging.info("Assigning identities")
    list_of_fragments.compute_P2_vectors()
    fragment = list_of_fragments.get_next_fragment_to_identify()
    while fragment:
        fragment.assign_identity()
        fragment = list_of_fragments.get_next_fragment_to_identify()


def assign_remaining_fragments(
    list_of_fragments: ListOfFragments,
    identification_model: nn.Module,
    network_params: NetworkParams,
):
    """This is the main function of this module: given a list_of_fragments it
    puts in place the routine to identify, if possible, each of the individual
    fragments. The starting point for the identification is given by the
    predictions produced by the ConvNetwork net passed as input. The organisation
    of the images in individual fragments is then used to assign more accurately.

    Parameters
    ----------
    list_of_fragments : <ListOfFragments object>
        collection of the individual fragments and associated methods
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    net : <ConvNetwork object>
        Convolutional neural network object created according to net.params

    See Also
    --------
    ListOfFragments.get_images_from_fragments_to_assign
    assign
    compute_identification_statistics_for_non_accumulated_fragments

    """
    logging.info("Assigning identities to all non-accumulated individual fragments")
    list_of_fragments.reset(roll_back_to="accumulation")
    number_of_unidentified_individual_fragments = (
        list_of_fragments.get_number_of_unidentified_individual_fragments()
    )
    logging.info(
        "Number of unidentified individual fragments: "
        f"{number_of_unidentified_individual_fragments}"
    )
    if not number_of_unidentified_individual_fragments:
        list_of_fragments.compute_P2_vectors()
        return

    images = list_of_fragments.get_images_from_fragments_to_assign()

    predictions, softmax_probs = get_predictions_identities(
        identification_model, images, network_params
    )

    logging.debug(
        f"{len(predictions)} generated predictions between "
        f"identities {set(predictions)}"
    )
    compute_identification_statistics_for_non_accumulated_fragments(
        list_of_fragments.fragments, predictions, softmax_probs
    )
    assign_identity(list_of_fragments)
