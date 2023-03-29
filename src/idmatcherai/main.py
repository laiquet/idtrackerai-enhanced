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

plt.rcParams["font.family"] = "STIXgeneral"


def IdMatcherAi(folders: list[Path]):
    logging.info(
        "Matching sessions:\n    "
        + "\n    ".join(map(str, folders[1:]))
        + f"\nwith {folders[0]}"
    )
    master_session = Video.load(folders[0])

    for matching_session in map(Video.load, folders[1:]):
        logging.info("\nMatching %s", matching_session)
        if matching_session.number_of_animals != master_session.number_of_animals:
            logging.warning(
                "Different number of animals between\n   "
                f" {matching_session} ({matching_session.number_of_animals})"
                " and\n   "
                f" {master_session} ({master_session.number_of_animals})"
            )

        if matching_session.version != master_session.version:
            logging.warning(
                "Different idtracker.ai versions between\n    "
                f"{matching_session} {matching_session.id_image_size}"
                " and\n    "
                f"{master_session} {master_session.id_image_size}\n"
                "This can cause poor matchings"
            )

        if matching_session.id_image_size != master_session.id_image_size:
            logging.error(
                "Different identification image size between\n    "
                f"{matching_session} {matching_session.id_image_size}"
                " and\n    "
                f"{master_session} {master_session.id_image_size}\n"
                "Check how to define a fixed identification image size in"
                " http://idtracker.ai/en/latest/user_guide/usage.html#identification-image-size"
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
        save_matrix(
            direct_matching_mat,
            results_path,
            "direct_matches",
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        indirect_matching_mat = match(
            master_session.id_images_folder, matching_session.accumulation_folder
        )
        save_matrix(
            indirect_matching_mat.T,
            results_path,
            "indirect_matches",
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        joined_matching_mat = direct_matching_mat + indirect_matching_mat.T
        save_matrix(
            joined_matching_mat,
            results_path,
            "joined_matches",
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        assigned_ids, assignements = linear_sum_assignment(
            joined_matching_mat, maximize=True
        )
        assigned_ids += 1
        assignements += 1

        accuracy = (
            joined_matching_mat[assigned_ids - 1, assignements - 1].sum()
            / joined_matching_mat[assigned_ids - 1].sum()
        )
        save_matrix(
            joined_matching_mat,
            results_path,
            "joined_matches",
            (assignements, assigned_ids, accuracy),
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        with (results_path / "results.csv").open("w", encoding="utf_8") as file:
            for identity, assignment in zip(assigned_ids, assignements):
                file.write(f"{identity:2d}, {assignment:2d}\n")

        logging.info("Results in %s", results_path)
        logging.log(
            logging.INFO if accuracy > 0.8 else logging.WARNING,
            f"Matching accuracy: {accuracy:.2%}",
        )


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
    mat: np.ndarray,
    dir: Path,
    name: str,
    assign: tuple[np.ndarray, np.ndarray, float] | None = None,
    xlabel: str = "",
    ylabel: str = "",
):
    np.savetxt(
        (dir / "csv" / name).with_suffix(".csv"),
        mat,
        "%5d" if mat.shape[1] < 20 else "%d",
        delimiter=",",
    )
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(
        mat,
        interpolation="none",
        extent=(+0.5, mat.shape[1] + 0.5, mat.shape[0] + 0.5, +0.5),
    )

    ax.set(
        title=name.replace("_", " ").capitalize(),
        xlabel=xlabel,
        ylabel=ylabel,
        aspect="auto",
    )
    if assign is not None:
        ax.plot(assign[0], assign[1], "rx", ms=8, label="Assignment")
        ax.legend()
        ax.set_title(ax.get_title() + f" | Assignment accuracy: {assign[2]:.2%}")

    # show grid
    ax.set_xticks(np.arange(1.5, mat.shape[1]), minor=True)
    ax.set_yticks(np.arange(1.5, mat.shape[0]), minor=True)
    ax.grid(which="minor", color="w")
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im).set_label("Number of matches")
    fig.tight_layout(pad=0.8)
    fig.savefig(str((dir / "png" / name).with_suffix(".png")))
