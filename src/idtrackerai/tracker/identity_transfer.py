import logging

import numpy as np
from torch.nn import Module

from idtrackerai import Fragment, GlobalFragment, Video
from idtrackerai.network import fc_weights_reinit
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
):
    if (
        identification_model is not None and video.identity_transfer
    ):  # identity transfer
        logging.info(f"Transferring identities from {video.knowledge_transfer_folder}")
        identities = get_transferred_identities(
            first_global_fragment_for_accumulation, video, identification_model
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
            identities = np.arange(video.n_animals)
        else:
            logging.info(
                "[green bold]Identities transferred successfully!",
                extra={"markup": True},
            )
    else:
        logging.info(
            "Tracking without identity transfer, assigning random initial identities"
        )
        identities = np.arange(video.n_animals)

    for id, fragment in zip(identities, first_global_fragment_for_accumulation):
        fragment.acceptable_for_training = True
        fragment.temporary_id = id
        frequencies = np.zeros(video.n_animals)
        frequencies[id] = fragment.n_images
        fragment.certainty = 1.0
        fragment.set_P1_from_frequencies(frequencies)


def get_transferred_identities(
    first_global_fragment_for_accumulation: GlobalFragment,
    video: Video,
    identification_model: Module,
) -> list[int | None] | None:
    images, _ = first_global_fragment_for_accumulation.get_images_and_labels(
        video.id_images_file_paths
    )

    predictions, softmax_probs = get_predictions_identities(
        identification_model, images
    )

    compute_identification_statistics_for_non_accumulated_fragments(
        first_global_fragment_for_accumulation.fragments,
        predictions,
        softmax_probs,
        video.n_animals,
    )

    # Check certainties of the individual fragments in the global fragment
    # for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:

    for fragment in first_global_fragment_for_accumulation:
        fragment.acceptable_for_training = True

    for fragment in first_global_fragment_for_accumulation:
        if fragment.certainty < conf.CERTAINTY_THRESHOLD:
            logging.error(
                "A fragment is not certain enough, "
                f"CERTAINTY_THRESHOLD = {conf.CERTAINTY_THRESHOLD:.2f}, "
                f"fragment certainty = {fragment.certainty:.2f}"
            )
            return None

    P1_array, index_individual_fragments_sorted_by_P1 = get_P1_array_and_argsort(
        first_global_fragment_for_accumulation
    )

    # assign temporary identity to individual fragments by hierarchical P1
    for fragment_indx in index_individual_fragments_sorted_by_P1:
        fragment: Fragment = first_global_fragment_for_accumulation.fragments[
            fragment_indx
        ]

        if p1_below_random(P1_array, fragment_indx, fragment):
            logging.error("The computed identities P1 is below random")
            return None

        temporary_id = int(np.argmax(P1_array[fragment_indx]))
        if fragment.is_inconsistent_with_coexistent_fragments(temporary_id):
            logging.error("The computed identities are not consistent")
            return None
        P1_array = set_fragment_temporary_id(
            fragment, temporary_id, P1_array, fragment_indx
        )

    # Check if the global fragment is unique after assigning the identities
    if not first_global_fragment_for_accumulation.is_unique(video.n_animals):
        logging.error("The computed identities are not unique")
        return None

    return [
        fragment.temporary_id for fragment in first_global_fragment_for_accumulation
    ]
