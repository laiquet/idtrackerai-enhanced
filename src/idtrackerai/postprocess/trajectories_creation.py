import logging
from typing import Iterable

import numpy as np

from idtrackerai import Blob, ListOfBlobs, ListOfFragments, Video
from idtrackerai.utils import create_dir

from .assign_them_all import close_trajectories_gaps
from .compute_velocity_model import compute_model_velocity
from .correct_impossible_velocity_jumps import correct_impossible_velocity_jumps
from .get_trajectories import produce_output_dict
from .trajectories_to_csv import convert_trajectories_file_to_csv_and_json


def trajectories_API(
    video: Video,
    list_of_blobs: ListOfBlobs,
    single_global_fragment: bool,
    list_of_fragments: ListOfFragments,
):
    if (
        not video.track_wo_identities
        and not video.single_animal
        and not single_global_fragment
    ):
        postprocess_impossible_jumps(video, list_of_fragments, list_of_blobs.all_blobs)

    video.create_trajectories_timer.start()
    create_dir(video.trajectories_folder, remove_existing=True)

    trajectories = produce_output_dict(
        list_of_blobs.blobs_in_video, video, list_of_fragments.fragments
    )
    list_of_blobs.save(video.blobs_path)

    trajectories_file = video.trajectories_folder / "with_gaps.npy"

    logging.info(f"Saving trajectories with gaps in {trajectories_file}")
    np.save(trajectories_file, trajectories)  # type: ignore
    if video.convert_trajectories_to_csv_and_json:
        convert_trajectories_file_to_csv_and_json(
            trajectories_file, video.add_time_column_to_csv
        )

    if (
        not video.track_wo_identities
        and not video.single_animal
        and not single_global_fragment
    ):
        interpolate_crossings(video, list_of_blobs, list_of_fragments)
    else:
        video.estimated_accuracy = 1.0
    video.create_trajectories_timer.finish()
    video.general_timer.finish()
    video.save()


def postprocess_impossible_jumps(
    video: Video, list_of_fragments: ListOfFragments, all_blobs: Iterable[Blob]
):
    video.impossible_jumps_timer.start()
    video.velocity_threshold = compute_model_velocity(list_of_fragments)
    correct_impossible_velocity_jumps(video, list_of_fragments)

    video.individual_fragments_stats = list_of_fragments.get_stats()

    video.estimated_accuracy = compute_estimated_accuracy(list_of_fragments)
    list_of_fragments.save(video.fragments_path)
    list_of_fragments.update_blobs(all_blobs)
    video.impossible_jumps_timer.finish()


def compute_estimated_accuracy(list_of_fragments: ListOfFragments) -> float:
    weighted_P2 = 0
    number_of_individual_blobs = 0

    for fragment in list_of_fragments.individual_fragments:
        if fragment.assigned_identities[0] not in (0, None):
            assert fragment.P2_vector is not None
            weighted_P2 += (
                fragment.P2_vector[fragment.assigned_identities[0] - 1]
                * fragment.n_images
            )
        number_of_individual_blobs += fragment.n_images
    return weighted_P2 / number_of_individual_blobs


def interpolate_crossings(
    video: Video, list_of_blobs_gaps: ListOfBlobs, list_of_fragments: ListOfFragments
):
    video.crossing_solver_timer.start()
    list_of_blobs_no_gaps = ListOfBlobs.load(video.blobs_path)
    close_trajectories_gaps(video, list_of_blobs_no_gaps, list_of_fragments)
    list_of_blobs_no_gaps.save(video.blobs_no_gaps_path)
    for blob in list_of_blobs_no_gaps.all_blobs:
        # save some RAM
        blob.centroid
        blob.area
        del blob.contour
        if hasattr(blob, "convexHull"):
            del blob.convexHull
    video.crossing_solver_timer.finish()

    trajectories_wo_gaps_file = video.trajectories_folder / "without_gaps.npy"
    logging.info(
        "Generating trajectories. The trajectories files are stored in "
        f"{trajectories_wo_gaps_file}"
    )
    trajectories_wo_gaps = produce_output_dict(
        list_of_blobs_no_gaps.blobs_in_video, video, list_of_fragments.fragments
    )

    np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)  # type: ignore
    if video.convert_trajectories_to_csv_and_json:
        convert_trajectories_file_to_csv_and_json(
            trajectories_wo_gaps_file, video.add_time_column_to_csv
        )

    assign_zeros_with_interpolation_identities(
        list_of_blobs_gaps, list_of_blobs_no_gaps
    )
    list_of_blobs_gaps.save(video.blobs_path)
    logging.info("Saving improved trajectories with gaps")
    trajectories_file = video.trajectories_folder / "with_gaps.npy"
    trajectories = produce_output_dict(
        list_of_blobs_gaps.blobs_in_video, video, list_of_fragments.fragments
    )
    np.save(trajectories_file, trajectories)  # type: ignore
    if video.convert_trajectories_to_csv_and_json:
        convert_trajectories_file_to_csv_and_json(
            trajectories_file, video.add_time_column_to_csv
        )


def assign_zeros_with_interpolation_identities(
    list_of_blobs_gaps: ListOfBlobs, list_of_blobs_no_gaps: ListOfBlobs
):
    counter = 0
    for blobs_in_frame_gaps, blobs_in_frame_no_gaps in zip(
        list_of_blobs_gaps.blobs_in_video, list_of_blobs_no_gaps.blobs_in_video
    ):
        unassigned_blobs = [
            blob
            for blob in blobs_in_frame_gaps
            if blob.is_an_individual and blob.assigned_identities[0] == 0
        ]
        for unassigned_blob in unassigned_blobs:
            candidate_blobs = [
                blob
                for blob in blobs_in_frame_no_gaps
                if blob.fragment_identifier == unassigned_blob.fragment_identifier
            ]
            if (
                len(candidate_blobs) == 1
                and len(candidate_blobs[0].assigned_identities) == 1
            ):
                unassigned_blob.identities_corrected_closing_gaps = candidate_blobs[
                    0
                ].assigned_identities
                counter += 1
    logging.debug(
        f"Corrected {counter} individual blob identities in ListOfBlobs with gaps found"
        " during closing gaps"
    )
