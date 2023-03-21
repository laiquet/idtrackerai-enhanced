"""
# TODO: Think better the structure of folders and where to save.
Maybe VideoFolders instead of session folders.
Maybe better a list with two session folders or a list of lists of session folders
Think what escalates better
# TODO: Think what happens when num animals is different
# TODO: Comment
"""
import logging
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
from pprint import pprint

import numpy as np
import toml

from idtrackerai import Video
from idtrackerai.utils import conf, initLogger

from .images import extact_all_images_and_labels
from .matcher import get_transfer_dicts, joined_results, match
from .summaries import print_summary_matching

# from network import train_model_from


class IdMatcherAi:
    """Class to match identities between idtracker.ai sessions and between folders
    containing several idtracker.ai session for single animal videos.
    """

    def __init__(
        self, folderA: Path, folderB: Path, plot_results_flag=False, rematch_ids=False
    ):
        """Initializes the class

        Parameters
        ----------
        folderA : str
            Path to the session folder of a video tracked with idtracker.ai or
            path to a folder containing multiple session folders from single
            animal videos. In the second case, the folder name cannot
            contain the name 'session'
        folderB :  str
            same as folderA
        save_folder : str
            Path to the folder where to save the results of the matching
        plot_results : bool
            If True plots a summary of the matching
        """
        self.rematch_ids = rematch_ids
        self.folderA = folderA
        self.folderB = folderB

        self.video_A = Video.load(self.folderA)
        self.video_B = Video.load(self.folderB)

        self.plot_results_flag = plot_results_flag

    def match_identities(self):
        """matches the identities between the two folders"""

        self.matching_results_file_A = (
            self.folderA / "matching_results" / (self.folderB.name + "-.npy")
        )
        self.matching_results_file_B = (
            self.folderB / "matching_results" / (self.folderA.name + "-.npy")
        )
        self.matching_results_file_A.parent.mkdir(exist_ok=True)
        self.matching_results_file_B.parent.mkdir(exist_ok=True)

        self.modelA = self.video_A.accumulation_folder
        self.modelB = self.video_B.accumulation_folder

        self.imagesA, self.labelsA = extact_all_images_and_labels(
            self.video_A.id_images_folder.glob("id_images_*.hdf5")
        )
        self.imagesB, self.labelsB = extact_all_images_and_labels(
            self.video_B.id_images_folder.glob("id_images_*.hdf5")
        )

        self.get_matching_results()

        np.save(self.matching_results_file_A, self.matching_results)
        np.save(self.matching_results_file_B, self.matching_results)

        # single_animals images and video
        # print("Matching from two sets of single animal videos")
        # raise NotImplementedError
        # self.get_models_folders()
        # self.get_images_and_labels()
        # self.get_matching_results()

    # def train_model(self, folder):
    #     """Train model"""
    #     print("Training model for {}".format(folder))
    #     identification_images_paths = [
    #         p
    #         for p in glob.glob(os.path.join(folder, "*"))
    #         if "identification_images" in p
    #     ]
    #     sessions_paths = [
    #         p for p in glob.glob(os.path.join(folder, "*")) if "session" in p
    #     ]
    #     if len(identification_images_paths) == 1:
    #         identification_images_file_path = (
    #             identification_images_file_path
    #         ) = os.path.join(identification_images_paths[0], "i_images.hdf5")
    #         if os.path.isfile(identification_images_file_path):
    #             return train_model_from(identification_images_file_path)
    #         else:
    #             raise ValueError(
    #                 "No file found in {}".format(identification_images_paths[0])
    #             )
    #     elif len(sessions_paths) > 1:
    #         print("Multiple session folders found in {}".format(folder))
    #         identification_images_file_paths = [
    #             os.path.join(s, "identification_images", "i_images.hdf5")
    #             for s in sessions_paths
    #         ]
    #         if all([os.path.isfile(p) for p in identification_images_file_paths]):
    #             return train_model_from(identification_images_file_paths)
    #         else:
    #             raise ValueError(
    #                 "Some session folder in {} does not contain a i_imagees.hdf5 files".format(
    #                     folder
    #                 )
    #             )

    """ Matching """

    def get_matching_results(self):
        # match model A to images and labels in B
        logging.info(
            "Matching model %s to %s labeled images from %s",
            self.folderA,
            len(self.imagesB),
            self.folderB,
        )
        confusion_matrix, frequencies_matrix, _ = match(
            self.modelA, self.imagesB, self.labelsB
        )
        transfer_dicts = get_transfer_dicts(confusion_matrix, frequencies_matrix)
        self.matching_results_A_B = {
            "network_from": self.folderA,
            "images_from": self.folderB,
            "P1_confusion_matrix": confusion_matrix,
            "frequencies_matrix": frequencies_matrix,
            "transfer_dicts": transfer_dicts,
        }

        print_summary_matching(transfer_dicts, self.folderB, self.folderA)

        # match model B to images and labels in A
        logging.info(
            "Matching model %s to %s labeled images from %s",
            self.modelB,
            len(self.imagesA),
            self.folderA,
        )
        confusion_matrix, frequencies_matrix, _ = match(
            self.modelB, self.imagesA, self.labelsA
        )
        transfer_dicts = get_transfer_dicts(confusion_matrix, frequencies_matrix)
        self.matching_results_B_A = {
            "network_from": self.folderB,
            "images_from": self.folderA,
            "P1_confusion_matrix": confusion_matrix,
            "frequencies_matrix": frequencies_matrix,
            "transfer_dicts": transfer_dicts,
        }

        print_summary_matching(transfer_dicts, self.folderA, self.folderB)
        exit()
        # joined results
        results = joined_results(self.matching_results_A_B, self.matching_results_B_A)
        self.matching_results = {
            "matching_results_A_B": self.matching_results_A_B,
            "matching_results_B_A": self.matching_results_B_A,
            "matching_results": results,
        }

        pprint(results["matches_dict_separated"]["hungarian_freq"])
        pprint(results["matches_dict_joined"]["hungarian_freq"])


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
        "sessionA",
        help="path to the session folder with the results from the first video",
        type=path,
    )
    parser.add_argument(
        "sessionB",
        help="path to the session folder with the results from the second video",
        type=path,
    )
    parser.add_argument(
        "--save_folder",
        "-sf",
        help="folder where to save the results",
        type=path,
        default=Path.cwd(),
    )
    args = parser.parse_args()

    matcher = IdMatcherAi(args.sessionA, args.sessionB, args.save_folder)

    matcher.match_identities()
