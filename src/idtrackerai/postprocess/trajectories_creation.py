import logging

import numpy as np

from idtrackerai import (
    Blob,
    Fragment,
    ListOfBlobs,
    ListOfFragments,
    ListOfGlobalFragments,
    Video,
)
from idtrackerai.utils import conf, create_dir

from .assign_them_all import close_trajectories_gaps
from .compute_velocity_model import compute_model_velocity
from .correct_impossible_velocity_jumps import (
    correct_impossible_velocity_jumps,
)
from .get_trajectories import produce_output_dict
from .identify_non_assigned_with_interpolation import (
    assign_zeros_with_interpolation_identities,
)
from .trajectories_to_csv import convert_trajectories_file_to_csv_and_json


def trajectories_API(
    video: Video,
    list_of_blobs: ListOfBlobs,
    list_of_global_fragments: ListOfGlobalFragments,
    list_of_fragments: ListOfFragments,
):

    if (
        not video.track_wo_identities
        and not video.single_animal
        and not list_of_global_fragments.single_global_fragment
    ):
        postprocess_impossible_jumps(
            video, list_of_fragments, list_of_blobs.blobs_in_video
        )

    video.create_trajectories_timer.start()
    create_dir(video.trajectories_folder)

    if not video.track_wo_identities:
        trajectories_file = video.trajectories_folder / "trajectories.npy"
        trajectories = produce_output_dict(
            list_of_blobs.blobs_in_video,
            video,
        )
    else:
        trajectories_file = (
            video.trajectories_folder / "trajectories_wo_identification.npy"
        )
        trajectories = produce_output_dict(
            list_of_blobs.blobs_in_video,
            video,
        )
    logging.info("Saving trajectories")
    np.save(trajectories_file, trajectories)  # type: ignore
    if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
        logging.info("Saving trajectories in csv format...")
        convert_trajectories_file_to_csv_and_json(trajectories_file)

    video._has_trajectories = True

    if (
        not video.track_wo_identities
        and not video.single_animal
        and not list_of_global_fragments.single_global_fragment
    ):
        interpolate_crossings(video, list_of_blobs, list_of_fragments)
    else:
        video.estimated_accuracy = 1.0
        video._has_trajectories_wo_gaps = False
    video.create_trajectories_timer.finish()


def postprocess_impossible_jumps(
    video: Video,
    list_of_fragments: ListOfFragments,
    blobs_in_video: list[list[Blob]],
):
    video.impossible_jumps_timer.start()
    video.velocity_threshold = compute_model_velocity(
        list_of_fragments.fragments
    )
    correct_impossible_velocity_jumps(video, list_of_fragments)

    video.individual_fragments_stats = list_of_fragments.get_stats()

    video.estimated_accuracy = compute_estimated_accuracy(
        list_of_fragments.fragments
    )
    list_of_fragments.save(
        video.accumulation_folder / "list_of_fragments.pickle"
    )
    list_of_fragments.update_blobs(blobs_in_video)
    video.impossible_jumps_timer.finish()


def compute_estimated_accuracy(fragments: list[Fragment]) -> float:
    weighted_P2 = 0
    number_of_individual_blobs = 0

    for fragment in fragments:
        if fragment.is_an_individual:
            if fragment.assigned_identities[0] not in (0, None):
                weighted_P2 += (
                    fragment.P2_vector[fragment.assigned_identities[0] - 1]
                    * fragment.number_of_images
                )
            number_of_individual_blobs += fragment.number_of_images
    return weighted_P2 / number_of_individual_blobs


def interpolate_crossings(
    video: Video,
    list_of_blobs: ListOfBlobs,
    list_of_fragments: ListOfFragments,
):
    video.crossing_solver_timer.start()
    list_of_blobs_no_gaps = list_of_blobs.get_deep_copy()
    list_of_blobs_no_gaps = close_trajectories_gaps(
        video,
        list_of_blobs_no_gaps,
        list_of_fragments,
    )
    list_of_blobs_no_gaps.save(video.blobs_no_gaps_path)
    video.crossing_solver_timer.finish()
    trajectories_wo_gaps_file = (
        video.trajectories_folder / "trajectories_wo_gaps.npy"
    )
    logging.info(
        "Generating trajectories. The trajectories files are stored in "
        f"{trajectories_wo_gaps_file}"
    )
    trajectories_wo_gaps = produce_output_dict(
        list_of_blobs_no_gaps.blobs_in_video,
        video,
    )
    np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)  # type: ignore
    if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
        logging.info("Saving trajectories in csv format...")
        convert_trajectories_file_to_csv_and_json(trajectories_wo_gaps_file)
    video._has_trajectories_wo_gaps = True
    logging.info("Saving trajectories")
    list_of_blobs = assign_zeros_with_interpolation_identities(
        list_of_blobs,
        list_of_blobs_no_gaps,
    )
    trajectories_file = video.trajectories_folder / "trajectories.npy"
    trajectories = produce_output_dict(
        list_of_blobs.blobs_in_video,
        video,
    )
    np.save(trajectories_file, trajectories)  # type: ignore
    if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
        logging.info("Saving trajectories in csv format...")
        convert_trajectories_file_to_csv_and_json(trajectories_file)
