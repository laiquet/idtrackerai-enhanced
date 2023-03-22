"""
# TODO: Think what happens when num animals is different
# TODO: Comment
"""
import pickle
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import toml

from idtrackerai import Video
from idtrackerai.utils import conf, create_dir, initLogger

from .matcher import get_transfer_dicts, joined_results, match


def IdMatcherAi(folders: list[Path]):
    master_session = Video.load(folders[0])

    for matching_session in map(Video.load, folders[1:]):
        results_path = matching_session.idmatcher_results_path / (
            master_session.session_folder.name
        )
        create_dir(results_path)

        direct_confusion_mat, direct_frequencies_mat, _certainties = match(
            matching_session.id_images_folder, master_session.accumulation_folder
        )
        draw_matrix(direct_confusion_mat, results_path, "direct_confusion")
        draw_matrix(direct_frequencies_mat, results_path, "direct_frequencies")

        direct_matching_results = {
            "network_from": str(master_session.accumulation_folder),
            "images_from": str(matching_session.id_images_folder),
            "P1_confusion_matrix": direct_confusion_mat,
            "frequencies_matrix": direct_frequencies_mat,
            "transfer_dicts": get_transfer_dicts(
                direct_confusion_mat, direct_frequencies_mat
            ),
        }

        indirect_confusion_mat, indirect_frequencies_mat, _certainties = match(
            master_session.id_images_folder, matching_session.accumulation_folder
        )
        draw_matrix(indirect_confusion_mat, results_path, "indirect_confusion")
        draw_matrix(indirect_frequencies_mat, results_path, "indirect_frequencies")

        indirect_matching_results = {
            "network_from": str(matching_session.accumulation_folder),
            "images_from": str(matching_session.id_images_folder),
            "P1_confusion_matrix": indirect_confusion_mat,
            "frequencies_matrix": indirect_frequencies_mat,
            "transfer_dicts": get_transfer_dicts(
                direct_confusion_mat, indirect_frequencies_mat
            ),
        }

        matching_results = joined_results(
            direct_matching_results, indirect_matching_results
        )

        pprint(matching_results["matches_dict_separated"]["hungarian_freq"])
        pprint(matching_results["matches_dict_joined"]["hungarian_freq"])
        matching_results = {
            "direct_results": direct_matching_results,
            "indirect_results": indirect_matching_results,
            "joined_results": matching_results,
        }

        results_path = matching_session.idmatcher_results_path / (
            master_session.session_folder.name
        )
        results_path.parent.mkdir(exist_ok=True)

        with open(results_path.with_suffix(".pickle"), "wb") as file:
            pickle.dump(matching_results, file)


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
