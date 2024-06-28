import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from idtrackerai import Fragment, GlobalFragment, Session
from idtrackerai.utils import IdtrackeraiError, conf

from ..network import IdentifierBase, get_predictions, load_identifier_model
from .accumulation_manager import (
    get_P1_array_and_argsort,
    p1_below_random,
    set_fragment_temporary_id,
)
from .assigner import compute_identification_statistics_for_non_accumulated_fragments


def identify_first_global_fragment_for_accumulation(
    first_global_fragment_for_accumulation: GlobalFragment,
    session: Session,
    knowledge_transfer_folder: Path | None = None,
    image_size: Sequence[int] | None = None,
):
    logging.info(
        "Using the Global Fragment starting at frame %d as the first one in"
        " accumulation",
        first_global_fragment_for_accumulation.first_frame_of_the_core,
    )

    if knowledge_transfer_folder:
        logging.info(f"Transferring identities from {knowledge_transfer_folder}")
        try:
            identity_transfer_model = load_identifier_model(
                knowledge_transfer_folder, image_size
            )
            identities = get_transferred_identities(
                first_global_fragment_for_accumulation, session, identity_transfer_model
            )
            logging.info("Identity transfer succeeded.")
            session.identity_transfer_succeded = True
        except Exception as exc:
            logging.error(
                "[red bold]Identity transfer failed[/]: %s", exc, extra={"markup": True}
            )
            identities = np.arange(session.n_animals)
            session.identity_transfer_succeded = False
    else:
        logging.info(
            "Tracking without identity transfer, assigning random initial identities"
        )
        identities = np.arange(session.n_animals)

    for id, fragment in zip(identities, first_global_fragment_for_accumulation):
        fragment.acceptable_for_training = True
        fragment.temporary_id = id
        frequencies = np.zeros(session.n_animals)
        frequencies[id] = fragment.n_images
        fragment.certainty = 1.0
        fragment.set_P1_from_frequencies(frequencies)


def get_transferred_identities(
    first_global_fragment_for_accumulation: GlobalFragment,
    session: Session,
    identification_model: IdentifierBase,
):
    images, _ = first_global_fragment_for_accumulation.get_images_and_labels()

    predictions, softmax_probs = get_predictions(
        identification_model, images, session.id_images_file_paths
    )

    compute_identification_statistics_for_non_accumulated_fragments(
        first_global_fragment_for_accumulation.fragments,
        predictions,
        softmax_probs,
        session.n_animals,
    )

    # Check certainties of the individual fragments in the global fragment
    # for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:

    for fragment in first_global_fragment_for_accumulation:
        fragment.acceptable_for_training = True

    for fragment in first_global_fragment_for_accumulation:
        if fragment.certainty < conf.CERTAINTY_THRESHOLD:
            raise IdtrackeraiError(
                "A fragment is not certain enough, "
                f"CERTAINTY_THRESHOLD = {conf.CERTAINTY_THRESHOLD:.2f}, "
                f"fragment certainty = {fragment.certainty:.2f}"
            )

    P1_array, index_individual_fragments_sorted_by_P1 = get_P1_array_and_argsort(
        first_global_fragment_for_accumulation
    )

    # assign temporary identity to individual fragments by hierarchical P1
    for fragment_indx in index_individual_fragments_sorted_by_P1:
        fragment: Fragment = first_global_fragment_for_accumulation.fragments[
            fragment_indx
        ]

        if p1_below_random(P1_array, fragment_indx, fragment):
            raise IdtrackeraiError("The computed identities P1 is below random")

        temporary_id = int(P1_array[fragment_indx].argmax())
        if fragment.is_inconsistent_with_coexistent_fragments(temporary_id):
            raise IdtrackeraiError("The computed identities are not consistent")
        P1_array = set_fragment_temporary_id(
            fragment, temporary_id, P1_array, fragment_indx
        )

    # Check if the global fragment is unique after assigning the identities
    if not first_global_fragment_for_accumulation.is_unique(session.n_animals):
        raise IdtrackeraiError("The computed identities are not unique")

    identities: list[int] = []

    for fragment in first_global_fragment_for_accumulation:
        if fragment.temporary_id is None:
            raise IdtrackeraiError("Not all fragments have been properly identified")
        identities.append(fragment.temporary_id)

    return identities
