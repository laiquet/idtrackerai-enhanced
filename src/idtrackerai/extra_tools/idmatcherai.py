"""Idmatcher.ai matches identities from two different tracked
sessions, see :ref:`idmatcher.ai`."""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from idtrackerai import ListOfFragments, Session
from idtrackerai.base.network import (
    IdentifierBase,
    get_predictions,
    load_identifier_model,
)
from idtrackerai.utils import create_dir, resolve_path, wrap_entrypoint

plt.rcParams["font.family"] = "STIXgeneral"


@wrap_entrypoint
def idmatcherai_entrypoint() -> None:
    for handler in logging.root.handlers:
        handler.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "sessions",
        help="path to the session folder with the results from the first video",
        type=path,
        nargs="+",
    )
    args = parser.parse_args()

    idmatcherai(*args.sessions)


def idmatcherai(*folders: Path | str | Session) -> None:
    """idmatcherai script, called by the command ``idmatcherai``.


    .. seealso::
        Documentation for :ref:`idmatcher.ai`

    Parameters
    ----------
    *folders : Path | str | Session
        Sequence of Sessions or session paths to match with idmatcherai. With two elements, the matching is performed between them. With more than two elements, the matching will be computed between the first and the second, the first and the third, the first and the fourth, and so on. If the paths to the Sessions are given, :meth:`idtrackerai.Session.load` will be used to load them.
    """
    logging.info(
        "Matching sessions:\n    "
        + "\n    ".join(map(str, folders[1:]))
        + f"\nwith {folders[0]}"
    )
    master_session = folders[0]
    if not isinstance(master_session, Session):
        master_session = Session.load(master_session)

    master_fragments = ListOfFragments.load(
        master_session.fragments_path, reconnect=False
    )

    for matching_session in folders[1:]:
        if not isinstance(matching_session, Session):
            matching_session = Session.load(matching_session)

        logging.info("\nMatching %s", matching_session)
        if matching_session.n_animals != master_session.n_animals:
            logging.warning(
                "Different number of animals between\n   "
                f" {matching_session} ({matching_session.n_animals})"
                " and\n   "
                f" {master_session} ({master_session.n_animals})"
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
                " https://idtracker.ai/latest/user_guide/usage.html#knowledge-and-identity-transfer"
            )
            continue

        results_path = matching_session.idmatcher_results_path / (
            master_session.session_folder.name
        )
        create_dir(results_path)
        create_dir(results_path / "csv")
        create_dir(results_path / "png")

        direct_matches = match(
            ListOfFragments.load(matching_session.fragments_path, reconnect=False),
            matching_session.id_images_file_paths,
            master_session.accumulation_folder,
        )
        save_matrix(
            direct_matches,
            results_path,
            "direct_matches",
            xlabel=master_session.session_folder.name,
            ylabel=matching_session.session_folder.name,
        )

        indirect_matches = match(
            master_fragments,
            master_session.id_images_file_paths,
            matching_session.accumulation_folder,
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

        logging.info(f"{master_session.n_animals} animals in {master_session}")
        logging.info(f"{matching_session.n_animals} animals in {matching_session}")

        if matching_session.n_animals == master_session.n_animals:
            logging.info("Maximizing direct and indirect matches")
            matrix_to_optimize = joined_matches
        elif matching_session.n_animals > master_session.n_animals:
            logging.info("Maximizing indirect matches only")
            matrix_to_optimize = indirect_matches
        elif matching_session.n_animals < master_session.n_animals:
            logging.info("Maximizing direct matches only")
            matrix_to_optimize = direct_matches
        else:
            raise RuntimeError(matching_session.n_animals, master_session.n_animals)

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
                    f"{assigned_id + 1:8d}, {assignments[i] + 1:10d},"
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


def match(
    fragments: ListOfFragments, id_images_paths: list[Path], model_path: Path
) -> np.ndarray:

    if not model_path.exists() and model_path.with_name("accumulation_0").exists():
        # backward compatibility
        model_path = model_path.with_name("accumulation_0")

    logging.info(
        "Matching images from %s with model from %s",
        id_images_paths[0].parent,
        model_path,
    )

    image_locations_list: list[tuple[int, int]] = []
    labels_list: list[int] = []
    for fragment in fragments.individual_fragments:
        if fragment.identity not in (None, 0):
            image_locations_list += fragment.image_locations
            labels_list += [fragment.identity] * len(fragment)
    labels = np.asarray(labels_list)
    image_locations = np.asarray(image_locations_list)
    image_locations_list.clear()  # clear the list to save memory
    labels_list.clear()  # clear the list to save memory

    max_images = 10000 * fragments.n_animals
    n_images = len(image_locations)
    if n_images > max_images:
        logging.info(
            "Too many images (%d), randomly sampling 10000 images per animal", n_images
        )
        sample = np.random.choice(n_images, max_images, replace=False)
        image_locations = image_locations[sample]
        labels = labels[sample]

    # we sort the images to speedup the loading process
    order = np.lexsort(image_locations.T)
    image_locations = image_locations[order]
    labels = labels[order]

    model, model_n_classes = load_identification_model(model_path)
    all_predictions = get_predictions(model, image_locations, id_images_paths)[0]

    set_of_labels = np.unique(labels)

    n_img_ids = len(set_of_labels)
    """number of labels in the images to be assigned by the model"""

    n_model_ids = model_n_classes
    """number of classes in the model"""

    confusion_matrix = np.zeros((n_img_ids, n_model_ids), int)

    for identity in set_of_labels:
        predictions = all_predictions[labels == identity]
        confusion_matrix[identity - 1] = np.bincount(
            predictions, minlength=n_model_ids + 1
        )[1:]
    return confusion_matrix


def load_identification_model(model_folder: Path) -> tuple[IdentifierBase, int]:
    params_path = model_folder / "model_params.json"
    if params_path.is_file():
        params = json.loads(params_path.read_text())
    elif params_path.with_suffix(".npy").is_file():
        params = np.load(params_path.with_suffix(".npy"), allow_pickle=True).item()
    else:
        raise FileNotFoundError(params_path)

    n_classes = (  # 5.1.6 compatibility
        params["n_classes"] if "n_classes" in params else params["number_of_classes"]
    )

    identification_model = load_identifier_model(model_folder, params["image_size"])
    return identification_model, n_classes


def score_row(row: np.ndarray, assigned) -> float:
    major_indices = row.argsort()[::-1]
    major_value = row[major_indices[0]]
    if major_value == 0:
        return 0
    for major_competitor in major_indices:
        if major_competitor != assigned:
            return float((row[assigned] - row[major_competitor]) / major_value)
    raise ValueError


def path(value: str) -> Path:
    return_path = resolve_path(value)
    if not return_path.exists():
        raise ValueError()
    return return_path


def save_matrix(
    mat: np.ndarray,
    dir: Path,
    name: str,
    assign: tuple[np.ndarray, np.ndarray, float] | None = None,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
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

    # avoid confusions
    if name.startswith("direct"):
        xlabel = "Predictions by model from " + xlabel
        ylabel = "Images from " + ylabel
    elif name.startswith("indirect"):
        xlabel = "Images from " + xlabel
        ylabel = "Predictions by model from " + ylabel

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
    plt.close()
