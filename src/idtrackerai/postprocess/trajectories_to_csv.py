# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from idtrackerai.utils import create_dir, wrap_exceptions


def save_array_to_csv(path: Path, array: np.ndarray, key: str):
    array = np.squeeze(array)
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
    np.savetxt(path, array, delimiter=",", header=array_header, fmt=fmt, comments="")


def convert_trajectories_file_to_csv_and_json(npy_path: Path):
    logging.info(f"Converting {npy_path} to .csv and .json")
    output_dir = npy_path.with_suffix("")
    create_dir(output_dir, remove_existing=True)
    try:
        trajectories_dict: dict = np.load(npy_path, allow_pickle=True).item()
        attributes_dict = {}
        for key, value in trajectories_dict.items():
            if key in ("trajectories", "id_probabilities"):
                save_array_to_csv(
                    output_dir / npy_path.with_suffix(f".{key}.csv").name,
                    value,
                    key=key,
                )
            elif key == "areas":
                np.savetxt(
                    output_dir / npy_path.with_suffix(f".{key}.csv").name,
                    np.asarray((value["mean"], value["median"], value["std"])).T,
                    delimiter=",",
                    header="mean, median, standard_deviation",
                    fmt="%.1f",
                    comments="",
                )
            else:
                attributes_dict[key] = value

        json_path = output_dir / npy_path.with_suffix(".attributes.json").name
        json.dump(attributes_dict, json_path.open("w"), indent=4)
    except Exception as e:
        logging.error(e)


@wrap_exceptions
def main():
    logging.basicConfig(level=logging.DEBUG, format="%(message)s", datefmt="%H:%M:%S")

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

    args = parser.parse_args()

    for path in args.paths:
        if path.is_file() and path.suffix == ".npy":
            convert_trajectories_file_to_csv_and_json(path)

        if path.name.startswith("session_"):
            path /= "trajectories"

        for file in path.glob("*.npy"):
            convert_trajectories_file_to_csv_and_json(file)


if __name__ == "__main__":
    main()
