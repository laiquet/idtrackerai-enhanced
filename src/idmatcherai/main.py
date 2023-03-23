"""
# TODO: Think what happens when num animals is different
# TODO: Comment
"""
import json
import logging
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml
from scipy.optimize import linear_sum_assignment

from idtrackerai import Video
from idtrackerai.utils import conf, create_dir, initLogger

from .matcher import match


def IdMatcherAi(folders: list[Path]):
    master_session = Video.load(folders[0])

    for matching_session in map(Video.load, folders[1:]):
        results_path = matching_session.idmatcher_results_path / (
            master_session.session_folder.name
        )
        create_dir(results_path)
        create_dir(results_path / "csv")
        create_dir(results_path / "png")

        direct_confusion_mat, direct_frequencies_mat = match(
            matching_session.id_images_folder, master_session.accumulation_folder
        )
        save_matrix(direct_confusion_mat, results_path, "direct_confusion")
        save_matrix(direct_frequencies_mat, results_path, "direct_frequencies")

        indirect_confusion_mat, indirect_frequencies_mat = match(
            master_session.id_images_folder, matching_session.accumulation_folder
        )
        save_matrix(indirect_confusion_mat, results_path, "indirect_confusion")
        save_matrix(indirect_frequencies_mat, results_path, "indirect_frequencies")

        joined_frequencies_mat = direct_frequencies_mat + indirect_frequencies_mat.T
        joined_confusion_mat = 1.0 - (1.0 - direct_confusion_mat) * (
            1.0 - indirect_confusion_mat.T
        )
        save_matrix(joined_confusion_mat, results_path, "joined_confusion")
        save_matrix(joined_frequencies_mat, results_path, "joined_frequencies")

        joined_assing_P1 = (
            linear_sum_assignment(joined_confusion_mat, maximize=True)[1] + 1
        )
        joined_assing_freq = (
            linear_sum_assignment(joined_frequencies_mat, maximize=True)[1] + 1
        )

        save_matrix(
            joined_confusion_mat, results_path, "joined_confusion", joined_assing_freq
        )
        save_matrix(
            joined_frequencies_mat,
            results_path,
            "joined_frequencies",
            joined_assing_freq,
        )

        with open(results_path.with_suffix(".json"), "w") as file:
            json.dump(
                {
                    "joined_assing_P1": joined_assing_P1.tolist(),
                    "joined_assing_freq": joined_assing_freq.tolist(),
                },
                file,
            )
        with open(results_path.with_suffix(".toml"), "w") as file:
            file.write(f"joined_assing_P1 = {joined_assing_P1.tolist()}\n")
            file.write(f"joined_assing_freq = {joined_assing_freq.tolist()}\n")


def defaults() -> dict:
    toml_dict = toml.load((files("idtrackerai") / "constants.toml").open())

    for key, value in toml_dict.items():
        if value == "":
            toml_dict[key] = None

    return toml_dict


def path(value: str):
    return_path = Path(value).expanduser().resolve()
    if not return_path.exists():
        raise ValueError()
    return return_path


def main():
    initLogger(level=logging.INFO)
    conf.set_dict(defaults())

    parser = ArgumentParser()
    parser.add_argument(
        "sessions",
        help="path to the session folder with the results from the first video",
        type=path,
        nargs="+",
    )
    args = parser.parse_args()

    IdMatcherAi(args.sessions)


def save_matrix(
    mat: np.ndarray, dir: Path, name: str, assign: np.ndarray | None = None
):
    np.savetxt(
        (dir / "csv" / name).with_suffix(".csv"),
        mat,
        "%5d" if issubclass(mat.dtype.type, np.integer) else "%7.5f",
        delimiter=",",
    )
    fig, ax = plt.subplots()
    im = ax.imshow(
        mat,
        interpolation="none",
        extent=(+0.5, mat.shape[0] + 0.5, mat.shape[1] + 0.5, +0.5),
    )
    ax.set(title=name.replace("_", " "))
    if assign is not None:
        ax.plot(assign, range(1, len(assign) + 1), "r.", ms=8)
    fig.colorbar(im)
    fig.tight_layout(pad=0.3)
    fig.savefig(str((dir / "png" / name).with_suffix(".png")))
