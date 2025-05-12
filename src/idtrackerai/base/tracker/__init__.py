import gc
import logging
from itertools import pairwise

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
        track_without_identities(session, list_of_blobs)
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


def track_without_identities(session: Session, list_of_blobs: ListOfBlobs):
    logging.info("Tracking without identities")
    session.number_of_animals = max(map(len, list_of_blobs.blobs_in_video))

    current_fragments = [-10 for _ in range(session.number_of_animals)]

    for blobs_in_frame, blobs_in_future in pairwise(
        track(list_of_blobs.blobs_in_video, "Assigning random identities")
    ):
        next_fragments = {b.fragment_identifier for b in blobs_in_future}

        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                continue
            try:
                identity = current_fragments.index(blob.fragment_identifier)
            except ValueError:  # blob's fragment is not in current_fragments
                identity = current_fragments.index(-10)  # look for an empty spot
                current_fragments[identity] = blob.fragment_identifier

            blob.identity = identity + 1
            if blob.fragment_identifier not in next_fragments:
                current_fragments[identity] = -10  # leave an empty spot

    for blob in list_of_blobs.blobs_in_video[-1]:  # last frame
        if blob.is_a_crossing:
            continue
        try:
            identity = current_fragments.index(blob.fragment_identifier)
        except ValueError:  # blob.fragment_identifier is not in identifiers_prev
            identity = current_fragments.index(-10)  # look for an empty spot
            current_fragments[identity] = blob.fragment_identifier
        blob.identity = identity + 1
