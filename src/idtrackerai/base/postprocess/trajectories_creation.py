import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import numpy as np

from idtrackerai import Blob, ListOfBlobs, ListOfFragments, Session
from idtrackerai.utils import create_dir, json_default, resolve_path, wrap_entrypoint

from .assign_them_all import close_trajectories_gaps
from .compute_velocity_model import compute_model_velocity
from .correct_impossible_jumps import correct_impossible_velocity_jumps
from .get_trajectories import produce_output_dict


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
        with session.new_timer("Impossible jumps correction"):
            postprocess_impossible_jumps(
                session, list_of_fragments, list_of_blobs.all_blobs
            )

    if session.track_wo_identities or session.single_animal or single_global_fragment:
        session.estimated_accuracy = 1.0
        return

    with session.new_timer("Crossings solver"):
        close_trajectories_gaps(session, list_of_blobs, list_of_fragments)

    trajectories_path = session.trajectories_folder / "trajectories.npy"
    create_dir(session.trajectories_folder, remove_existing=True)
    logging.info(f"Generating trajectories in {trajectories_path}")
    trajectories = produce_output_dict(
        list_of_blobs.blobs_in_video, session, list_of_fragments.fragments
    )
    np.save(trajectories_path, trajectories)  # type: ignore
    if session.convert_trajectories_to_csv_and_json:
        try:
            convert_trajectories_file_to_csv_and_json(
                trajectories_path, session.add_time_column_to_csv
            )
        except Exception as exc:
            # Do not crash if the trajectory conversion failed
            logging.error(exc)


def postprocess_impossible_jumps(
    session: Session, list_of_fragments: ListOfFragments, all_blobs: Iterable[Blob]
):
    session.velocity_threshold = compute_model_velocity(list_of_fragments)
    correct_impossible_velocity_jumps(session, list_of_fragments)

    session.individual_fragments_stats = list_of_fragments.get_stats()

    session.estimated_accuracy = compute_estimated_accuracy(list_of_fragments)
    list_of_fragments.save(session.fragments_path)
    list_of_fragments.update_blobs(all_blobs)


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


def save_array_to_csv(path: Path, array: np.ndarray, key: str, fps: float | None):
    array = array.squeeze()
    if key == "id_probabilities":
        fmt = "%.3e"
    elif key == "trajectories":
        fmt = "%.3f"
    else:
        fmt = "%.3f"

    if array.ndim == 3:
        array_header = ",".join(
            coord + str(i) for i in range(1, array.shape[1] + 1) for coord in ("x", "y")
        )
        array = array.reshape((-1, array.shape[1] * array.shape[2]))
    elif array.ndim == 2:
        array_header = ",".join(f"{key}{i}" for i in range(1, array.shape[1] + 1))
    else:
        raise ValueError(array.shape)

    fmt = [fmt] * array.shape[1]

    if fps is not None:  # add time column
        array_header = "seconds," + array_header
        fmt = ["%.3f"] + fmt
        time = np.arange(len(array), dtype=float) / fps
        array = np.column_stack((time, array))

    np.savetxt(path, array, delimiter=",", header=array_header, fmt=fmt, comments="")


def convert_trajectories_file_to_csv_and_json(
    npy_path: Path, add_time_column: bool = False
):
    output_dir = npy_path.parent / (npy_path.stem + "_csv")
    create_dir(output_dir, remove_existing=True)

    logging.info(f"Converting {npy_path} to .csv and .json")
    trajectories_dict: dict = np.load(npy_path, allow_pickle=True).item()
    attributes_dict = {}
    for key, value in trajectories_dict.items():
        if key in ("trajectories", "id_probabilities"):
            save_array_to_csv(
                output_dir / (key + ".csv"),
                value,
                key=key,
                fps=(
                    trajectories_dict.get("frames_per_second", 1)
                    if add_time_column
                    else None
                ),
            )
        elif key == "areas":
            np.savetxt(
                output_dir / (key + ".csv"),
                np.asarray((value["mean"], value["median"], value["std"])).T,
                delimiter=",",
                header="mean, median, standard_deviation",
                fmt="%.1f",
                comments="",
            )
        else:
            attributes_dict[key] = value

    json.dump(
        attributes_dict,
        (output_dir / "attributes.json").open("w"),
        indent=4,
        default=json_default,
    )


@wrap_entrypoint
def main():
    """Script to convert trajectory formats"""
    parser = ArgumentParser()

    parser.add_argument(
        "paths",
        help=(
            "Paths to convert trajectories to CSV and JSON. Can be session folders (to"
            " convert all .npy files inside trajectory subfolder), arbitrary folder (to"
            " convert all .npy files in it) and specific .npy files."
        ),
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "--add_time",
        help="Add a time column (in seconds) to csv trajectory files.",
        action="store_true",
    )

    args = parser.parse_args()

    for path in args.paths:
        path = resolve_path(path)
        if not path.exists():
            logging.warning('Path "%s" not found', path)
            continue
        files_found = False
        if path.is_file() and path.suffix == ".npy":
            convert_trajectories_file_to_csv_and_json(path, args.add_time)
            files_found = True

        if path.name.startswith("session_"):
            path /= "trajectories"

        for file in path.glob("*.npy"):
            convert_trajectories_file_to_csv_and_json(file, args.add_time)
            files_found = True

        if not files_found:
            logging.warning('No trajectory files found in "%s"', path)


if __name__ == "__main__":
    main()
