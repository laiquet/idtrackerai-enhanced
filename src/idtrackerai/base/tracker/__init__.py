import gc
import logging
from itertools import pairwise
from pprint import pformat

import numpy as np

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Session
from idtrackerai.base.network import IdentifierBase
from idtrackerai.utils import track


def tracker_API(
    session: Session,
    list_of_blobs: ListOfBlobs,
    list_of_fragments: ListOfFragments,
    list_of_global_fragments: ListOfGlobalFragments,
) -> IdentifierBase | None:

    if session.track_wo_identities:
        track_without_identities(session, list_of_blobs, list_of_fragments)
        return None

    if session.single_animal:
        track_single_animal(list_of_blobs)
        return None

    if len(list_of_fragments) == 1:
        logging.warning("Tracking a single fragment")
        for blob in list_of_blobs.all_blobs:
            if blob.fragment_identifier == list_of_fragments.fragments[0].identifier:
                blob.identity = 1
        return None

    from .tracker import run_tracker

    logging.info(
        "Deleting ListOfBlobs to save memory, it will be reloaded from disk after"
        " tracking"
    )
    for blob in list_of_blobs.all_blobs:
        blob.__dict__.clear()
    list_of_blobs.blobs_in_video.clear()
    # Blobs contain circular references between them, so the automatic garbage
    # collector won't delete them immediately after the clear().
    # Manually calling gc.collect() is the way to really free RAM
    gc.collect()
    identifier_model = run_tracker(session, list_of_fragments, list_of_global_fragments)
    list_of_fragments.update_id_images_dataset()
    gc.collect()  # just in case
    list_of_blobs.blobs_in_video = ListOfBlobs.load(session.blobs_path).blobs_in_video
    return identifier_model


def track_single_animal(list_of_blobs: ListOfBlobs):
    logging.info("Tracking a single animal, assigning identity 1 to all blobs")
    for blob in list_of_blobs.all_blobs:
        blob.identity = 1


def track_without_identities(
    session: Session, list_of_blobs: ListOfBlobs, list_of_fragments: ListOfFragments
):
    logging.info("Tracking without identities")
    roi_to_ids: dict[int, set[int]] = {}
    if session.exclusive_rois:  # Assign identities based on exclusive ROIs if present
        all_rois = {frag.exclusive_roi for frag in list_of_fragments}
        n_blobs_per_region = np.zeros(
            (list_of_blobs.number_of_frames, max(all_rois) + 1), dtype=int
        )

        for fragment in list_of_fragments.individual_fragments:
            n_blobs_per_region[
                fragment.start_frame : fragment.end_frame, fragment.exclusive_roi
            ] += 1

        max_blobs_per_region = n_blobs_per_region.max(axis=0)
        counter = 0
        for roi, max_blobs in enumerate(max_blobs_per_region):
            roi_to_ids[roi] = set(range(counter, counter + max_blobs))
            counter += max_blobs

        session.number_of_animals = max_blobs_per_region.sum()
        session.identities_groups = {
            f"Region_{roi}": {id_ + 1 for id_ in ids} for roi, ids in roi_to_ids.items()
        }
        logging.info(
            "Identity groups by exclusive ROIs:\n%s", pformat(session.identities_groups)
        )
    else:
        session.number_of_animals = max(
            len(frame) for frame in list_of_blobs.blobs_in_video
        )
        roi_to_ids[-1] = set(range(session.number_of_animals))

    # Initialize current fragments for each animal slot
    current_fragments = [-10 for _ in range(session.number_of_animals)]

    # Iterate through frames in pairs to assign identities
    for blobs_in_frame, blobs_in_next in pairwise(
        track(list_of_blobs.blobs_in_video + [[]], "Assigning random identities")
    ):
        next_fragments = {b.fragment_identifier for b in blobs_in_next}

        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                continue

            try:
                identity = current_fragments.index(blob.fragment_identifier)
            except ValueError:
                # Find a valid identity slot for this fragment
                roi = (
                    list_of_fragments.fragments[blob.fragment_identifier].exclusive_roi
                    if session.exclusive_rois
                    else -1
                )
                valid_ids = roi_to_ids[roi]
                possible_ids = [
                    idx
                    for idx, frag_id in enumerate(current_fragments)
                    if frag_id == -10 and idx in valid_ids
                ]
                if not possible_ids:
                    logging.error(
                        f"No valid identity found for fragment {blob.fragment_identifier} in ROI {roi}"
                    )
                    continue
                identity = possible_ids[0]
                current_fragments[identity] = blob.fragment_identifier

            blob.identity = identity + 1
            if blob.fragment_identifier not in next_fragments:
                current_fragments[identity] = -10  # Free the slot
