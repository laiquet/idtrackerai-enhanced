"""
# TODO: Think what happens when num animals is different
# TODO: Comment
"""
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
    logging.info(
        "Matching sessions:\n    "
        + "\n    ".join(map(str, folders[1:]))
        + f"\nwith {folders[0]}"
    )
    master_session = Video.load(folders[0])

    for matching_session in map(Video.load, folders[1:]):
        if matching_session.number_of_animals != master_session.number_of_animals:
            logging.error(
                "Different number of animals between\n   "
                f" {matching_session} ({matching_session.number_of_animals})"
                " and\n   "
                f" {master_session} ({master_session.number_of_animals})"
            )
            continue

        if matching_session.id_image_size != master_session.id_image_size:
            logging.error(
                "Different identification image size between\n    "
                f"{matching_session} {matching_session.id_image_size}"
                " and\n    "
                f"{master_session} {master_session.id_image_size}"
            )
            continue
        results_path = matching_session.idmatcher_results_path / (
            master_session.session_folder.name
        )
        create_dir(results_path)
        create_dir(results_path / "csv")
        create_dir(results_path / "png")

        direct_matching_mat = match(
            matching_session.id_images_folder, master_session.accumulation_folder
        )
        save_matrix(direct_matching_mat, results_path, "direct_matches")

        indirect_matching_mat = match(
            master_session.id_images_folder, matching_session.accumulation_folder
        )
        save_matrix(indirect_matching_mat, results_path, "indirect_matches")

        joined_matching_mat = direct_matching_mat + indirect_matching_mat.T
        save_matrix(joined_matching_mat, results_path, "joined_matches")

        assignements = linear_sum_assignment(joined_matching_mat, maximize=True)[1] + 1

        save_matrix(joined_matching_mat, results_path, "joined_matches", assignements)

        np.savetxt(results_path.with_name("results.csv"), assignements, fmt="%d")

        accuracy = (
            joined_matching_mat[range(len(joined_matching_mat)), assignements - 1].sum()
            / joined_matching_mat.sum()
        )

        logging.info(f"Matching accuracy: {accuracy:.2%}")


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
    np.savetxt((dir / "csv" / name).with_suffix(".csv"), mat, "%5d", delimiter=",")
    fig, ax = plt.subplots()
    im = ax.imshow(
        mat,
        interpolation="none",
        extent=(+0.5, mat.shape[0] + 0.5, mat.shape[1] + 0.5, +0.5),
    )
    if assign is not None:
        ax.plot(assign, range(1, len(assign) + 1), "r.", ms=8)

    ax.set(title=name.replace("_", " ").capitalize())
    ax.xaxis.tick_top()

    # show grid
    ax.set_xticks(np.arange(1.5, mat.shape[0]), minor=True)
    ax.set_yticks(np.arange(1.5, mat.shape[1]), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    fig.colorbar(im).set_label("Number of matches")
    fig.tight_layout(pad=0.3)
    fig.savefig(str((dir / "png" / name).with_suffix(".png")), dpi=250)
