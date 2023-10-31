import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from idtrackerai.utils import create_dir, json_default, resolve_path, wrap_entrypoint


def save_array_to_csv(path: Path, array: np.ndarray, key: str, fps=float | None):
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
    npy_path: Path, add_time_column: bool = False, raise_errors=False
):
    output_dir = npy_path.parent / (npy_path.stem + "_csv")
    create_dir(output_dir, remove_existing=True)
    try:
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
    except Exception as e:
        if raise_errors:
            raise e
        logging.error(e)


@wrap_entrypoint
def main():
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
            convert_trajectories_file_to_csv_and_json(
                path, args.add_time, raise_errors=True
            )
            files_found = True

        if path.name.startswith("session_"):
            path /= "trajectories"

        for file in path.glob("*.npy"):
            convert_trajectories_file_to_csv_and_json(
                file, args.add_time, raise_errors=True
            )
            files_found = True

        if not files_found:
            logging.warning('No trajectory files found in "%s"', path)


if __name__ == "__main__":
    main()
