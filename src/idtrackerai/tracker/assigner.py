"""Identification of individual fragments given the predictions generate by the idCNN
"""
import logging
from shutil import copyfile

import numpy as np
from torch import load, nn

from idtrackerai import Fragment, ListOfFragments
from idtrackerai.network import NetworkParams
from idtrackerai.utils import Timer

from .identity_network import get_predictions_identities


def compute_identification_statistics_for_non_accumulated_fragments(
    fragments: list[Fragment],
    all_predictions: np.ndarray,
    all_softmax_probs: np.ndarray,
    number_of_animals: int,
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
                predictions, softmax_probs, number_of_animals
            )
            counter = next_counter_value


def check_penultimate_model(
    identification_model: nn.Module, network_params: NetworkParams
):
    """Loads the penultimate accumulation step if the validation accuracy of the last
    step was lower then the penultimate. This discard possible corrupt final accumulation steps
    """
    if not network_params.penultimate_model_path.is_file():
        logging.warning(
            "Penultimate model not found (%s)", network_params.penultimate_model_path
        )
        return

    last_accuracy = load(network_params.model_path).get("val_acc", 0.0)
    penultimate_model: dict = load(network_params.penultimate_model_path)
    penultimate_accuracy = penultimate_model.pop("val_acc", -1.0)
    logging.info(
        f"Last accuracy = {last_accuracy:.2%}, "
        f"Penultimate accuracy = {penultimate_accuracy:.2%}"
    )
    if penultimate_accuracy > last_accuracy:
        logging.info(
            "The last accumulation step had a lower accuracy than the penultimate."
        )
        logging.info(
            "Loading penultimate model, %s", network_params.penultimate_model_path
        )
        identification_model.load_state_dict(penultimate_model, strict=True)

        # set the penultimate as the one model
        network_params.model_path.unlink()
        copyfile(network_params.penultimate_model_path, network_params.model_path)
    else:
        logging.info(
            "The last accumulation step had a higher accuracy than the penultimate."
        )

    network_params.penultimate_model_path.unlink()


def assign_remaining_fragments(
    list_of_fragments: ListOfFragments,
    identification_model: nn.Module,
    network_params: NetworkParams,
    timer: Timer,
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
    timer.start()

    check_penultimate_model(identification_model, network_params)
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
        list_of_fragments.fragments,
        predictions,
        softmax_probs,
        list_of_fragments.number_of_animals,
    )

    logging.info("Assigning identities")
    list_of_fragments.compute_P2_vectors()
    fragment = list_of_fragments.get_next_fragment_to_identify()
    while fragment:
        fragment.assign_identity(list_of_fragments.number_of_animals)
        fragment = list_of_fragments.get_next_fragment_to_identify()

    timer.finish()
