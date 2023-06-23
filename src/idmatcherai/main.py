import logging
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml
from scipy.optimize import linear_sum_assignment

from idtrackerai import Video
from idtrackerai.utils import conf, create_dir, initLogger, wrap_exceptions

from .matcher import match

plt.rcParams["font.family"] = "STIXgeneral"


def IdMatcherAi(folders: list[Path]):
    conf.set_dict(defaults())
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
                f"{matching_session} {matching_session.version}"
                " and\n    "
                f"{master_session} {master_session.version}\n"
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

        direct_matches = match(
            matching_session.id_images_folder, master_session.auto_accumulation_folder
        )
        save_matrix(
            direct_matches,
            results_path,
            "direct_matches",
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        indirect_matches = match(
            master_session.id_images_folder, matching_session.auto_accumulation_folder
        ).T
        save_matrix(
            indirect_matches,
            results_path,
            "indirect_matches",
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        joined_matches = direct_matches + indirect_matches
        save_matrix(
            joined_matches,
            results_path,
            "joined_matches",
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        logging.info(f"{master_session.number_of_animals} animals in {master_session}")
        logging.info(
            f"{matching_session.number_of_animals} animals in {matching_session}"
        )

        if matching_session.number_of_animals == master_session.number_of_animals:
            logging.info("Maximizing direct and indirect matches")
            matrix_to_optimize = joined_matches
        elif matching_session.number_of_animals > master_session.number_of_animals:
            logging.info("Maximizing indirect matches only")
            matrix_to_optimize = indirect_matches
        elif matching_session.number_of_animals < master_session.number_of_animals:
            logging.info("Maximizing direct matches only")
            matrix_to_optimize = direct_matches
        else:
            RuntimeError(
                matching_session.number_of_animals, master_session.number_of_animals
            )

        logging.info("Assigning identities")
        assigned_ids, assignments = linear_sum_assignment(
            matrix_to_optimize, maximize=True
        )

        agreement = (
            joined_matches[assigned_ids, assignments].sum()
            / joined_matches[assigned_ids].sum()
        )

        direct_scores = [
            score_row(direct_matches[assigned_id], assignment)
            for assigned_id, assignment in zip(assigned_ids, assignments)
        ]

        indirect_scores = [
            score_row(indirect_matches[:, assignment], assigned_id)
            for assigned_id, assignment in zip(assigned_ids, assignments)
        ]

        with (results_path / "assignments.csv").open("w", encoding="utf_8") as file:
            file.write("identity, assignment, direct score, indirect score\n")
            for i, assigned_id in enumerate(assigned_ids):
                file.write(
                    f"{assigned_id+1:8d}, {assignments[i]+1:10d},"
                    f" {direct_scores[i]:12.4f}, {indirect_scores[i]:14.4f}\n"
                )

        mean_direct_score = float(np.mean(direct_scores))
        mean_indirect_score = float(np.mean(indirect_scores))
        mean_score = (mean_direct_score + mean_indirect_score) / 2
        save_matrix(
            direct_matches,
            results_path,
            "direct_matches",
            (assignments, assigned_ids, mean_direct_score),
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )
        save_matrix(
            indirect_matches,
            results_path,
            "indirect_matches",
            (assignments, assigned_ids, mean_indirect_score),
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )
        save_matrix(
            joined_matches,
            results_path,
            "joined_matches",
            (assignments, assigned_ids, mean_score),
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )
        logging.info("Results in %s", results_path)
        logging.info(
            "Scores:\n"
            f"    Matching agreement: {agreement:.2%}\n"
            f"    Mean direct score: {mean_direct_score:.2%}\n"
            f"    Mean indirect score: {mean_indirect_score:.2%}\n"
            f"    Total mean score: {mean_score:.2%}"
        )


def score_row(row: np.ndarray, assigned) -> float:
    major_indices = np.argsort(row)[::-1]
    major_value = row[major_indices[0]]
    if major_value == 0:
        return 0
    for major_competitor in major_indices:
        if major_competitor != assigned:
            return float((row[assigned] - row[major_competitor]) / major_value)
    raise ValueError


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


@wrap_exceptions
def main():
    initLogger(level=logging.INFO)

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
        ax.plot(assign[0] + 1, assign[1] + 1, "rx", ms=8, label="Assignment")
        ax.legend()
        direction = name.split("_")[0]
        ax.set_title(
            ax.get_title() + f" | Assignment {direction} score: {assign[2]:.2%}"
        )

    # show grid
    ax.set_xticks(np.arange(1.5, mat.shape[1]), minor=True)
    ax.set_yticks(np.arange(1.5, mat.shape[0]), minor=True)
    ax.grid(which="minor", color="w")
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im).set_label("Number of matches")
    fig.tight_layout(pad=0.8)
    fig.savefig(str((dir / "png" / name).with_suffix(".png")))
