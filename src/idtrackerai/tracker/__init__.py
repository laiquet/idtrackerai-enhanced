import logging
from itertools import pairwise

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Video
from idtrackerai.utils import IdtrackeraiError, track


def tracker_API(
    video: Video,
    list_of_blobs: ListOfBlobs,
    list_of_fragments: ListOfFragments,
    list_of_global_fragments: ListOfGlobalFragments,
) -> ListOfFragments:
    video.tracking_timer.start()

    if video.track_wo_identities:
        track_without_identities(video, list_of_blobs)

    elif video.single_animal:
        track_single_animal(list_of_blobs)

    elif list_of_global_fragments.no_global_fragment:
        raise IdtrackeraiError(
            "There are no Global Fragments long enough to be candidates"
            " for accumulation, thus it is not possible to train the"
            " identification networks. The video has to contain longer"
            " slices where all animals are visible without crossings."
        )

    elif list_of_global_fragments.single_global_fragment:
        track_single_global_fragment_video(
            video, list_of_blobs, list_of_fragments, list_of_global_fragments
        )

    else:
        from .tracker import TrackerAPI

        list_of_fragments = TrackerAPI(
            video, list_of_fragments, list_of_global_fragments
        ).track()
        list_of_fragments.update_id_images_dataset()

    video.tracking_timer.finish()
    return list_of_fragments


def track_single_global_fragment_video(
    video: Video,
    list_of_blobs: ListOfBlobs,
    list_of_fragments: ListOfFragments,
    list_of_global_fragments: ListOfGlobalFragments,
):
    logging.info("Tracking single global fragment")
    assert len(list_of_global_fragments.global_fragments) == 1
    global_fragment = list_of_global_fragments.global_fragments[0]

    for identity, fragment in enumerate(global_fragment):
        fragment.temporary_id = identity
        fragment.identity = identity + 1

    video.identities_groups = list_of_fragments.build_exclusive_rois()
    list_of_fragments.update_blobs(list_of_blobs.all_blobs)


def track_single_animal(list_of_blobs: ListOfBlobs):
    logging.info("Tracking a single animal, assigning identity 1 to all blobs")
    for blob in list_of_blobs.all_blobs:
        blob.identity = 1


def track_without_identities(video: Video, list_of_blobs: ListOfBlobs):
    logging.info("Tracking without identities")
    video.number_of_animals = list_of_blobs.max_number_of_blobs_in_one_frame

    current_fragments = [-10 for _ in range(video.number_of_animals)]

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
