import copy
import logging

import numpy as np

from idtrackerai import (
    ListOfBlobs,
    ListOfFragments,
    ListOfGlobalFragments,
    Video,
)
from idtrackerai.utils import conf

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
        and video.number_of_animals != 1
        and not list_of_global_fragments.single_global_fragment
    ):
        postprocess_impossible_jumps(video, list_of_fragments, list_of_blobs)

    video.create_trajectories_timer.start()

    if not video.track_wo_identities:
        video.create_trajectories_folder()
        trajectories_file = video.trajectories_folder / "trajectories.npy"
        trajectories = produce_output_dict(
            list_of_blobs.blobs_in_video,
            video,
        )
    else:
        video.create_trajectories_wo_identification_folder()
        trajectories_file = (
            video.trajectories_wo_identification_folder
            / "trajectories_wo_identification.npy"
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
        and video.number_of_animals != 1
        and not list_of_global_fragments.single_global_fragment
    ):
        interpolate_crossings(video, list_of_blobs, list_of_fragments)
    else:
        video.estimated_accuracy = 1.0
        video._has_crossings_solved = False
        video._has_trajectories_wo_gaps = False
        list_of_blobs.save(video.blobs_path)
    video.create_trajectories_timer.finish()


def postprocess_impossible_jumps(
    video: Video,
    list_of_fragments: ListOfFragments,
    list_of_blobs: ListOfBlobs,
):
    video.impossible_jumps_timer.start()
    video.velocity_threshold = compute_model_velocity(
        list_of_fragments.fragments
    )
    correct_impossible_velocity_jumps(video, list_of_fragments)

    video.individual_fragments_stats = list_of_fragments.get_stats()
    video.compute_estimated_accuracy(list_of_fragments.fragments)
    list_of_fragments.save(
        video.accumulation_folder / "list_of_fragments.pickle"
    )
    list_of_fragments.update_blobs(list_of_blobs.blobs_in_video)
    video.impossible_jumps_timer.finish()


def interpolate_crossings(
    video: Video,
    list_of_blobs: ListOfBlobs,
    list_of_fragments: ListOfFragments,
):

    list_of_blobs_no_gaps = copy.deepcopy(list_of_blobs)
    video._has_crossings_solved = False
    list_of_blobs_no_gaps = close_trajectories_gaps(
        video,
        list_of_blobs_no_gaps,
        list_of_fragments,
    )
    list_of_blobs_no_gaps.save(video.blobs_no_gaps_path)
    video._has_crossings_solved = True
    video.create_trajectories_wo_gaps_folder()
    logging.info(
        "Generating trajectories. The trajectories files are stored in %s"
        % video.trajectories_wo_gaps_folder
    )
    trajectories_wo_gaps_file = (
        video.trajectories_wo_gaps_folder / "trajectories_wo_gaps.npy"
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
    video.save()
