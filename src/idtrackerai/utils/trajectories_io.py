"Manage trajectory files saving and loading. In all compatible formats  and versions"

import json
import logging
import pickle
from argparse import ArgumentParser
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Literal

import numpy as np
from h5py import Dataset, File, Group

from . import create_dir, json_default, resolve_path, wrap_entrypoint


def save_trajectories(
    path: Path,
    data: dict,
    formats: Iterable[Literal["h5", "hdf5", "npy", "numpy", "csv", "json", "pickle"]],
) -> None:
    """Save trajectory dict into files following the format specifications"""
    for format in formats:
        format = format.lower()
        if format in ("npy", "numpy"):
            _save_trajectories_into_np(path, data)
        elif format in ("h5", "hdf5"):
            _save_trajectories_into_h5(path, data)
        elif format in ("csv", "json"):
            _save_trajectories_into_csv(path, data)
        elif format == "pickle":
            _save_trajectories_into_pickle(path, data)
        else:
            logging.error(f"Not recognized trajectory format {format!r}")


def _save_trajectories_into_np(path: Path, data: dict) -> None:
    path = path / "trajectories.npy"
    logging.info(f"Saving trajectories in {path}")
    np.save(path, data)  # type: ignore


def _save_trajectories_into_pickle(path: Path, data: dict) -> None:
    path = path / "trajectories.pickle"
    logging.info(f"Saving trajectories in {path}")
    with path.open("wb") as file:
        pickle.dump(data, file)


def _save_trajectories_into_csv(path: Path, data: dict) -> None:
    path = path / "trajectories_csv"
    logging.info(f"Saving trajectories in {path}")
    create_dir(path, remove_existing=True)

    attributes_dict = {}
    for key, value in data.items():
        if key in ("trajectories", "id_probabilities"):
            _save_array_to_csv(
                path / (key + ".csv"),
                value,
                key=key,
                fps=data.get("frames_per_second", 1),
            )
        elif key == "areas":
            np.savetxt(
                path / (key + ".csv"),
                np.asarray((value["mean"], value["median"], value["std"])).T,
                delimiter=", ",
                header="mean, median, standard_deviation",
                fmt="%.1f",
                comments="",
            )
        else:
            attributes_dict[key] = value

    json.dump(
        attributes_dict,
        (path / "attributes.json").open("w"),
        indent=4,
        default=json_default,
    )


def _save_trajectories_into_h5(path: Path, data: dict) -> None:
    path = path / "trajectories.h5"
    logging.info(f"Saving trajectories in {path}")
    data_ = data.copy()
    with File(path, "w") as file:
        areas: dict = data_.pop("areas")
        areas_group = file.create_group("areas")
        for key, value in areas.items():
            areas_group.create_dataset(key, data=value)

        setup_points: dict = data_.pop("setup_points")
        setup_points_group = file.create_group("setup_points")
        for key, value in setup_points.items():
            setup_points_group.create_dataset(key, data=value)

        identities_groups: dict = data_.pop("identities_groups")
        identities_groups_group = file.create_group("identities_groups")
        for key, value in identities_groups.items():
            identities_groups_group.create_dataset(key, data=np.asarray(value))

        # length_unit can be None
        file.attrs["length_unit"] = data_.pop("length_unit") or -1

        for key, value in data_.items():
            if value is None:
                continue
            try:
                if isinstance(value, (int, float, str)) or (
                    isinstance(value, list) and isinstance(value[0], str)
                ):
                    file.attrs[key] = value
                else:
                    compression = "gzip" if isinstance(value, np.ndarray) else None
                    file.create_dataset(key, data=value, compression=compression)
            except Exception as exc:
                logging.error(
                    f'Error saving "{key}" of type {type(value).__name__}. Error: {exc}'
                )


def _save_array_to_csv(
    path: Path, array: np.ndarray, key: str, fps: float | None
) -> None:
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
        array_header = f"{key}1"

    fmt = [fmt] * (array.shape[1] if array.ndim > 1 else 1)

    if fps is not None:  # add time column
        array_header = "time," + array_header
        fmt = ["%.3f"] + fmt
        time = np.arange(len(array), dtype=float) / fps
        array = np.column_stack((time, array))

    np.savetxt(path, array, delimiter=",", header=array_header, fmt=fmt, comments="")


def load_trajectories(path: Path | str) -> dict:
    """Loads trajectories from the specific path.

    Parameters
    ----------
    path : Path | str
        Path to trajectories. This can be a session folder, a trajectories
        folder or a specific file in any of the accepted file formats

    Returns
    -------
    dict
        Trajectories dict

    Raises
    ------
    FileNotFoundError
        If `path` does not exist
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    if path.is_file() or (path.is_dir() and path.name.endswith("_csv")):
        return _load_trajectories_file(path)

    if path.is_dir() and path.name.startswith("session_"):
        path = path / "trajectories"

    for name in (
        "trajectories",
        "validated",
        "without_gaps",
        "with_gaps",
        "trajectories_validated",
        "trajectories_wo_gaps",
        "trajectories_wo_identification",
    ):
        for format in (".h5", ".hdf5", ".npy", ".pickle", "_csv"):
            with suppress(FileNotFoundError):
                return _load_trajectories_file(path / (name + format))

    raise FileNotFoundError(f"Could not found trajectories in {path}")


def _load_trajectories_file(path: Path) -> dict:
    if path.suffix in (".h5", ".hdf5"):
        out = _load_trajectories_from_h5(path)
    elif path.suffix == ".pickle":
        out = _load_trajectories_from_pickle(path)
    elif path.suffix in (".npy", ".numpy"):
        out = _load_trajectories_from_np(path)
    elif path.is_dir() and path.name.endswith("_csv"):
        out = _load_trajectories_from_csv(path)
    else:
        raise FileNotFoundError(f'Could not load trajectory file "{path}"')
    logging.info(f"Trajectories loaded from {path}")
    return out


def _load_trajectories_from_np(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def _load_trajectories_from_pickle(path: Path) -> dict:
    with open(path, "rb") as file:
        return pickle.load(file)


def _load_trajectories_from_h5(path: Path) -> dict:
    def recursively_load_h5_group(group: Group) -> dict:
        ans = {}
        ans.update(group.attrs)

        for key, item in group.items():
            if isinstance(item, Dataset):
                ans[key] = item[()]
            elif isinstance(item, Group):
                ans[key] = recursively_load_h5_group(item)
            else:
                logging.error(f"Not recognized H5DF object '{key}': {item}")
        return ans

    with File(path, "r") as file:
        return recursively_load_h5_group(file)


def _load_trajectories_from_csv(path: Path) -> dict:
    out = json.loads((path / "attributes.json").read_text())

    for array_file in path.glob("*.csv"):
        # read csv without the first column containing time
        out[array_file.stem] = np.loadtxt(array_file, skiprows=1, delimiter=",")[:, 1:]

    if "trajectories" in out:
        # reshape trajectories
        out["trajectories"] = out["trajectories"].reshape(
            len(out["trajectories"]), -1, 2
        )

    return out


@wrap_entrypoint
def idtrackerai_format() -> None:
    """Script to convert trajectory formats"""
    parser = ArgumentParser()

    parser.add_argument(
        "paths",
        help=(
            "Paths to trajectory files to convert. Can be session folders "
            "(to convert all files inside trajectory subfolder), arbitrary folder "
            "(to convert all files in it) and specific files."
        ),
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "--formats",
        help="A sequence of strings defining in which formats the trajectories should be saved",
        type=str,
        choices=["h5", "npy", "csv", "pickle"],
        nargs="+",
        required=True,
    )

    args = parser.parse_args()

    for path in args.paths:
        path = resolve_path(path)
        if not path.exists():
            logging.error(f"{path} does not exist")
            continue

        trajectories = load_trajectories(path)

        if path.is_file():
            save_path = path.parent
        elif path.name.startswith("session_"):
            save_path = path / "trajectories"
        else:
            save_path = path

        save_trajectories(save_path, trajectories, args.formats)


if __name__ == "__main__":
    idtrackerai_format()
