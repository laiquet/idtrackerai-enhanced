import numpy as np

from idtrackerai import ListOfFragments
from idtrackerai.utils import conf, track


def compute_model_velocity(
    list_of_fragments: ListOfFragments, percentile=None
) -> float:
    """computes the 2 * (percentile) of the distribution of velocities of identified fish.
    params
    -----
    blobs_in_video: list of blob objects
        collection of blobs detected in the video.
    number_of_animals int
    percentile int
    -----
    return
    -----
    float
    2 * np.max(distance_travelled_in_individual_fragments) if percentile is None
    2 * percentile(velocity distribution of identified animals) otherwise
    """
    if percentile is None:
        percentile = conf.VEL_PERCENTILE

    distance_travelled_in_individual_frag: list[np.ndarray] = []
    for fragment in track(
        list_of_fragments.individual_fragments, "Computing velocity model"
    ):
        distance_travelled_in_individual_frag.append(fragment.frame_by_frame_velocity)

    distances = np.concatenate(distance_travelled_in_individual_frag)
    return (
        2 * distances.max()
        if percentile is None
        else 2 * float(np.percentile(distances, percentile, overwrite_input=True))
    )
