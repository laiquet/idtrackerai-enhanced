import logging

import numpy as np
from torch.nn import Module

from idtrackerai import GlobalFragment, Video
from idtrackerai.network import NetworkParams, fc_weights_reinit
from idtrackerai.utils import conf

from .accumulation_manager import (
    get_P1_array_and_argsort,
    p1_below_random,
    set_fragment_temporary_id,
)
from .assigner import compute_identification_statistics_for_non_accumulated_fragments
from .identity_network import get_predictions_identities


def identify_first_global_fragment_for_accumulation(
    first_global_fragment_for_accumulation: GlobalFragment,
    video: Video,
    identification_model: Module | None,
    network_params: NetworkParams,
    knowledge_transfer_info_dict: dict,
):
    if (
        identification_model is not None and video.identity_transfer
    ):  # identity transfer
        logging.info(f"Transferring identities from {video.knowledge_transfer_folder}")
        identities = get_transferred_identities(
            first_global_fragment_for_accumulation,
            video,
            identification_model,
            network_params,
            knowledge_transfer_info_dict,
        )

        if identities is None:
            logging.warning(
                "[red bold]Identity transfer failed", extra={"markup": True}
            )
            logging.info(
                "We proceed by reinitializing fully connected layers, "
                "assigning random identities to the first GlobalFragment "
                "and transferring only the convolutional filters "
                "(knowledge transfer)"
            )
            identification_model.apply(fc_weights_reinit)
            identities = np.arange(video.number_of_animals)
        else:
            logging.info(
                "[green bold] Identities transferred successfully",
                extra={"markup": True},
            )
    else:
        logging.info(
            "Tracking without identity transfer, assigning random initial identities"
        )
        identities = np.arange(video.number_of_animals)

    for id, fragment in zip(
        identities, first_global_fragment_for_accumulation.individual_fragments
    ):
        fragment.acceptable_for_training = True
        fragment.temporary_id = id
        frequencies = np.zeros(video.number_of_animals)
        frequencies[id] = fragment.number_of_images
        fragment.is_certain = True
        fragment.certainty = 1.0
        fragment.P1_vector = fragment.compute_P1_from_frequencies(frequencies)


def get_transferred_identities(
    first_global_fragment_for_accumulation: GlobalFragment,
    video: Video,
    identification_model: Module,
    network_params: NetworkParams,
    knowledge_transfer_info_dict: dict,
) -> list | None:
    images, _ = first_global_fragment_for_accumulation.get_images_and_labels(
        video.id_images_file_paths
    )

    predictions, softmax_probs = get_predictions_identities(
        identification_model, images, network_params
    )

    compute_identification_statistics_for_non_accumulated_fragments(
        first_global_fragment_for_accumulation.individual_fragments,
        predictions,
        softmax_probs,
        network_params.number_of_classes,
    )

    # Check certainties of the individual fragments in the global fragment
    # for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:

    for fragment in first_global_fragment_for_accumulation.individual_fragments:
        fragment.acceptable_for_training = True

    for fragment in first_global_fragment_for_accumulation.individual_fragments:
        if fragment.certainty < conf.CERTAINTY_THRESHOLD:
            logging.error(
                "A fragment is not certain enough, "
                f"CERTAINTY_THRESHOLD = {conf.CERTAINTY_THRESHOLD:.2f}, "
                f"fragment certainty = {fragment.certainty:.2f}"
            )
            return None

    P1_array, index_individual_fragments_sorted_by_P1_max_to_min = (
        get_P1_array_and_argsort(first_global_fragment_for_accumulation)
    )

    # assign temporary identity to individual fragments by hierarchical P1
    for fragment_indx in index_individual_fragments_sorted_by_P1_max_to_min:
        fragment = first_global_fragment_for_accumulation.individual_fragments[
            fragment_indx
        ]

        if p1_below_random(P1_array, fragment_indx, fragment):
            logging.error("The computed identities P1 is below random")
            return None

        temporary_id = int(np.argmax(P1_array[fragment_indx, :]))
        if not fragment.check_consistency_with_coexistent_individual_fragments(
            temporary_id
        ):
            logging.error("The computed identities are not consistent")
            return None
        P1_array = set_fragment_temporary_id(
            fragment, temporary_id, P1_array, fragment_indx
        )

    # Check if the global fragment is unique after assigning the identities
    if not first_global_fragment_for_accumulation.is_unique(video.number_of_animals):
        logging.error("The computed identities are not unique")
        return None

    if video.number_of_animals != knowledge_transfer_info_dict["number_of_classes"]:
        logging.error(
            "The number of animals in the current video and the one "
            "transferring identities from are not the same"
        )
        return None

    return [
        fragment.temporary_id
        for fragment in first_global_fragment_for_accumulation.individual_fragments
    ]
