"Identification of individual fragments given the predictions generate by the idCNN"

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from idtrackerai import Fragment, ListOfFragments
from idtrackerai.utils import track

from ..network import IdentifierBase, get_predictions


def compute_identification_statistics_for_non_accumulated_fragments(
    fragments: Sequence[Fragment],
    all_predictions: np.ndarray,
    all_probabilities: np.ndarray,
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
        next_counter_value = counter + fragment.n_images
        predictions = all_predictions[counter:next_counter_value]
        probabilities = all_probabilities[counter:next_counter_value]
        fragment.set_identification_statistics(
            predictions, probabilities, number_of_animals
        )
        counter = next_counter_value
    assert counter == len(all_predictions)


def check_penultimate_model(
    identification_model: torch.nn.Module,
    model_path: Path,
    penultimate_model_path: Path,
):
    """Loads the penultimate accumulation step if the validation accuracy of the last
    step was lower then the penultimate. This discard possible corrupt final accumulation steps
    """
    if not penultimate_model_path.is_file():
        return

    last_model: dict = json.loads(model_path.with_suffix(".metadata.json").read_text())
    last_accuracy = last_model.get("test_acc", 0.0)
    last_ratio_accumulated = last_model.pop("ratio_accumulated", 0.0)
    penultimate_model: dict = json.loads(
        penultimate_model_path.with_suffix(".metadata.json").read_text()
    )
    penultimate_accuracy = penultimate_model.pop("test_acc", -1.0)
    penultimate_ratio_accumulated = penultimate_model.pop("ratio_accumulated", -1.0)
    logging.info(
        f"Last accuracy = {last_accuracy:.2%}, "
        f"Last ratio accumulated = {last_ratio_accumulated:.2%}\n"
        f"Penultimate accuracy = {penultimate_accuracy:.2%}, "
        f"Penultimate ratio accumulated = {penultimate_ratio_accumulated:.2%}"
    )

    if penultimate_ratio_accumulated < 0.9:
        logging.info(
            "The penultimate accumulation step had a ration of accumulated images lower"
            " than 90%. Thus, it will be ignored."
        )
        return

    if penultimate_accuracy > last_accuracy:
        logging.info(
            "The last accumulation step had a lower accuracy than the penultimate."
        )
        logging.info("Loading penultimate model, %s", penultimate_model_path)
        identification_model.load_state_dict(
            torch.load(penultimate_model_path, weights_only=True)
        )
    else:
        logging.info(
            "The last accumulation step had a higher accuracy than the penultimate."
        )


def assign_remaining_fragments(
    list_of_fragments: ListOfFragments, identification_model: IdentifierBase
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
    session : <Session object>
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
    fragments_to_identify = [
        frag
        for frag in list_of_fragments.individual_fragments
        if not frag.used_for_training
    ]

    logging.info(
        f"Number of unidentified individual fragments: {len(fragments_to_identify)}"
    )
    if not fragments_to_identify:
        list_of_fragments.compute_P2_vectors()
        return

    image_locations: list[tuple[int, int]] = []
    for fragment in fragments_to_identify:
        image_locations += fragment.image_locations

    logging.info(
        "Number of images to identify non-accumulated fragments: %d",
        len(image_locations),
    )

    predictions, probabilities = get_predictions(
        identification_model, image_locations, list_of_fragments.id_images_file_paths
    )

    logging.debug(
        f"{len(predictions)} generated predictions between "
        f"{len(set(predictions))} identities"
    )
    compute_identification_statistics_for_non_accumulated_fragments(
        fragments_to_identify, predictions, probabilities, list_of_fragments.n_animals
    )

    list_of_fragments.compute_P2_vectors()
    for fragment in track(
        list_of_fragments.get_fragments_to_identify(),
        "Assigning remaining fragments identities",
        sum(frag.identity is None for frag in list_of_fragments.individual_fragments),
    ):
        fragment.assign_identity(
            list_of_fragments.n_animals, list_of_fragments.id_to_exclusive_roi
        )

    list_of_fragments.compute_P2_vectors()
