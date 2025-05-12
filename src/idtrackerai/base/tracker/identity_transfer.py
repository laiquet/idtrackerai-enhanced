import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

from idtrackerai import GlobalFragment, Session

from ..network import IdentifierBase, get_predictions, load_identifier_model


def identify_first_global_fragment_for_accumulation(
    initial_glob_frag: GlobalFragment,
    session: Session,
    knowledge_transfer_folder: Path | None = None,
    image_size: Sequence[int] | None = None,
):
    logging.info(
        "Using the Global Fragment at frame %d as the first one in accumulation",
        initial_glob_frag.first_frame_of_the_core,
    )

    if knowledge_transfer_folder:
        logging.info(f"Transferring identities from {knowledge_transfer_folder}")
        try:
            identity_transfer_model = load_identifier_model(
                knowledge_transfer_folder, image_size
            )
            identities = get_transferred_identities(
                initial_glob_frag, session.id_images_file_paths, identity_transfer_model
            )
            logging.info("[green]Identity transfer succeeded", extra={"markup": True})
            session.identity_transfer_succeeded = True
        except Exception as exc:
            logging.error(
                "[red bold]Identity transfer failed[/]: %s", exc, extra={"markup": True}
            )
            identities = np.arange(session.n_animals)
            session.identity_transfer_succeeded = False
    else:
        logging.info(
            "Tracking without identity transfer, assigning random initial identities"
        )
        identities = np.arange(session.n_animals)

    for id, fragment in zip(identities, initial_glob_frag):
        fragment.acceptable_for_training = True
        fragment.temporary_id = int(id)
        frequencies = np.zeros(session.n_animals)
        frequencies[id] = fragment.n_images
        fragment.certainty = 1.0
        fragment.set_P1_from_frequencies(frequencies)


def get_transferred_identities(
    initial_glob_frag: GlobalFragment,
    id_images_file_paths: Sequence[Path],
    identification_model: IdentifierBase,
) -> Sequence[int]:
    """Get the identities of the fragments in the initial global fragment using identity transfer."""

    images = []
    frag_id = []
    for i, fragment in enumerate(initial_glob_frag):
        images += fragment.image_locations
        frag_id += [i] * len(fragment)

    predictions, probabilities = get_predictions(
        identification_model, images, id_images_file_paths
    )

    confusion = confusion_matrix(frag_id, predictions - 1, normalize="true")
    frag_ids, identities = linear_sum_assignment(confusion, maximize=True)

    logging.info(
        f"Assigning identities in the {len(initial_glob_frag.fragments)} "
        "fragments of the initial global fragment from identity transfer:\n    "
        + "\n    ".join(
            f"Fragment #{initial_glob_frag.fragments[i].identifier} "
            f"({len(initial_glob_frag.fragments[i])} images long) assigned "
            f"to identity {j + 1} with {confusion[i, j] / np.sum(confusion[i]):.2%} agreement"
            for i, j in zip(frag_ids, identities)
        )
    )

    return identities
