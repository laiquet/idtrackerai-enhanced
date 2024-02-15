import logging
from typing import Iterable

import numpy as np

from idtrackerai import Blob, ListOfBlobs, ListOfFragments, Session
from idtrackerai.utils import create_dir

from .assign_them_all import close_trajectories_gaps
from .compute_velocity_model import compute_model_velocity
from .correct_impossible_jumps import correct_impossible_velocity_jumps
from .get_trajectories import produce_output_dict
from .trajectories_to_csv import convert_trajectories_file_to_csv_and_json


def trajectories_API(
    session: Session,
    list_of_blobs: ListOfBlobs,
    single_global_fragment: bool,
    list_of_fragments: ListOfFragments,
):
    if (
        not session.track_wo_identities
        and not session.single_animal
        and not single_global_fragment
    ):
        postprocess_impossible_jumps(
            session, list_of_fragments, list_of_blobs.all_blobs
        )

    session.create_trajectories_timer.start()
    create_dir(session.trajectories_folder, remove_existing=True)

    trajectories = produce_output_dict(
        list_of_blobs.blobs_in_video, session, list_of_fragments.fragments
    )

    trajectories_file = session.trajectories_folder / "with_gaps.npy"
    logging.info(f"Saving trajectories with gaps in {trajectories_file}")
    np.save(trajectories_file, trajectories)  # type: ignore
    if session.convert_trajectories_to_csv_and_json:
        convert_trajectories_file_to_csv_and_json(
            trajectories_file, session.add_time_column_to_csv
        )

    if (
        not session.track_wo_identities
        and not session.single_animal
        and not single_global_fragment
    ):
        interpolate_crossings(session, list_of_blobs, list_of_fragments)
    else:
        list_of_blobs.save(session.blobs_path)
        session.estimated_accuracy = 1.0
    session.create_trajectories_timer.finish()
    session.general_timer.finish()
    session.save()


def postprocess_impossible_jumps(
    session: Session, list_of_fragments: ListOfFragments, all_blobs: Iterable[Blob]
):
    session.impossible_jumps_timer.start()
    session.velocity_threshold = compute_model_velocity(list_of_fragments)
    correct_impossible_velocity_jumps(session, list_of_fragments)

    session.individual_fragments_stats = list_of_fragments.get_stats()

    session.estimated_accuracy = compute_estimated_accuracy(list_of_fragments)
    list_of_fragments.save(session.fragments_path)
    list_of_fragments.update_blobs(all_blobs)
    session.impossible_jumps_timer.finish()


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
    session: Session, list_of_blobs: ListOfBlobs, list_of_fragments: ListOfFragments
):
    close_trajectories_gaps(session, list_of_blobs, list_of_fragments)

    list_of_blobs.save(session.blobs_path)
    trajectories_wo_gaps_file = session.trajectories_folder / "without_gaps.npy"
    logging.info(
        "Generating trajectories. The trajectories files are stored in "
        f"{trajectories_wo_gaps_file}"
    )
    trajectories_wo_gaps = produce_output_dict(
        list_of_blobs.blobs_in_video, session, list_of_fragments.fragments
    )
    np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)  # type: ignore
    if session.convert_trajectories_to_csv_and_json:
        convert_trajectories_file_to_csv_and_json(
            trajectories_wo_gaps_file, session.add_time_column_to_csv
        )

    # reset crossings to save an improved version of with gaps
    for blob in list_of_blobs.all_blobs:
        if (
            blob.identities_corrected_closing_gaps is not None
            and len(blob.identities_corrected_closing_gaps) > 1
        ):
            blob.identities_corrected_closing_gaps = [0]

    trajectories_file = session.trajectories_folder / "with_gaps.npy"
    logging.info("Saving improved trajectories with gaps")
    trajectories = produce_output_dict(
        list_of_blobs.blobs_in_video, session, list_of_fragments.fragments
    )
    np.save(trajectories_file, trajectories)  # type: ignore
    if session.convert_trajectories_to_csv_and_json:
        convert_trajectories_file_to_csv_and_json(
            trajectories_file, session.add_time_column_to_csv
        )
