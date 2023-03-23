"""
# TODO: Think what happens when num animals is different
# TODO: Comment
"""
import json
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

        direct_confusion_mat, direct_frequencies_mat = match(
            matching_session.id_images_folder, master_session.accumulation_folder
        )
        draw_matrix(direct_confusion_mat, results_path, "direct_confusion")
        draw_matrix(direct_frequencies_mat, results_path, "direct_frequencies")

        indirect_confusion_mat, indirect_frequencies_mat = match(
            master_session.id_images_folder, matching_session.accumulation_folder
        )
        draw_matrix(indirect_confusion_mat, results_path, "indirect_confusion")
        draw_matrix(indirect_frequencies_mat, results_path, "indirect_frequencies")

        joined_frequencies_mat = direct_frequencies_mat + indirect_frequencies_mat.T
        joined_confusion_mat = 1.0 - (1.0 - direct_confusion_mat) * (
            1.0 - indirect_confusion_mat.T
        )

        joined_assing_P1 = linear_sum_assignment(joined_confusion_mat, maximize=True)[1]
        joined_assing_freq = linear_sum_assignment(
            joined_frequencies_mat, maximize=True
        )[1]

        with open(results_path.with_suffix(".json"), "w") as file:
            json.dump(
                {
                    "joined_assing_P1": joined_assing_P1,
                    "joined_assing_freq": joined_assing_freq,
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
    initLogger()
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


def draw_matrix(mat: np.ndarray, dir: Path, name: str):
    np.savetxt(
        (dir / name).with_suffix(".csv"),
        mat,
        "%5d" if issubclass(mat.dtype.type, np.integer) else "%7.5f",
        delimiter=",",
    )
    fig, ax = plt.subplots()
    im = ax.imshow(np.log(mat), interpolation="none")
    ax.set(title=name.replace("_", " "))
    fig.colorbar(im)
    fig.savefig(str((dir / name).with_suffix(".png")))
