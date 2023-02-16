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
import logging
from pathlib import Path

import numpy as np
from rich.progress import track
from torchvision.datasets.folder import VisionDataset

from idtrackerai import Blob
from idtrackerai.tracker.dataset.identification_dataset import duplicate_PCA_images
from idtrackerai.utils import conf, load_id_images


class CrossingDataset(VisionDataset):
    def __init__(
        self,
        blobs_list: list[Blob] | dict[str, list[Blob]],
        id_images_file_paths: list[Path],
        scope,
        transform=None,
    ):
        super().__init__(blobs_list, transform=transform)
        self.id_images_file_paths = id_images_file_paths
        self.blobs = blobs_list
        self.scope = scope
        self.images = None
        self.labels = None
        self.get_data()

    def get_data(self):
        if isinstance(self.blobs, dict):
            logging.info(f"Generating crossing {self.scope} set.")
            crossings_images = self.get_images_indices(image_type="crossings")
            crossing_labels = np.ones(len(crossings_images), int)

            logging.info(f"Generating single individual {self.scope} set")
            individual_images = self.get_images_indices(image_type="individuals")
            individual_labels = np.zeros(len(individual_images), int)

            logging.info("Preparing images and labels")
            images_indices = crossings_images + individual_images
            self.images = load_id_images(self.id_images_file_paths, images_indices)
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)

            self.labels = np.concatenate([crossing_labels, individual_labels], axis=0)

            if self.scope == "training":
                self.images, self.labels = duplicate_PCA_images(
                    self.images, self.labels
                )

            np.random.seed(0)
            permutation = np.random.permutation(len(self.labels))
            self.images = self.images[permutation]
            self.labels = self.labels[permutation]

        elif isinstance(self.blobs, list):
            images_indices = self.get_images_indices()
            self.images = load_id_images(self.id_images_file_paths, images_indices)
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)
            self.labels = np.zeros((self.images.shape[0]))

    def get_images_indices(self, image_type: str = "") -> list[tuple[int, int]]:
        if image_type:
            assert isinstance(self.blobs, dict)
            blobs = self.blobs[image_type]
        else:
            assert isinstance(self.blobs, list)
            blobs = self.blobs

        return [(blob.id_image_index, blob.episode) for blob in blobs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_train_validation_and_eval_blobs(
    blobs_in_video: list[list[Blob]],
    number_of_animals: int,
    ratio_validation: float = 0.1,
) -> tuple[dict[str, list[Blob]], dict[str, list[Blob]], list[Blob]]:
    """Given a list of blobs return 2 dictionaries (training_blobs, validation_blobs), and a list (toassign_blobs).

    :param list_of_blobs:
    :param ratio_validation:
    :return: training_blobs, validation_blobs, toassign_blobs
    """
    logging.info("Get list of blobs for training, validation and eval")

    individuals = []
    crossings = []
    toassign_blobs = []
    for blobs_in_frame in track(blobs_in_video, "First individual/crossing assignment"):
        in_a_global_fragment_core = len(blobs_in_frame) == number_of_animals
        for blob in blobs_in_frame:
            if in_a_global_fragment_core or blob.is_a_sure_individual():
                blob.used_for_training_crossings = True
                individuals.append(blob)
            elif blob.is_a_sure_crossing():
                blob.used_for_training_crossings = True
                crossings.append(blob)
            else:
                blob.used_for_training_crossings = False
                toassign_blobs.append(blob)

    logging.debug(
        f"{len(individuals)} individual, "
        f"{len(crossings)} crossing and "
        f"{len(toassign_blobs)} unknown blobs in total"
    )

    # Shuffle and make crossings and individuals even
    np.random.shuffle(individuals)
    np.random.shuffle(crossings)

    crossings = crossings[: conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR]
    individuals = individuals[: conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR]

    n_blobs_crossings = len(crossings)
    n_blobs_individuals = len(individuals)
    n_individual_blobs_validation = int(n_blobs_individuals * ratio_validation)
    n_crossing_blobs_validation = int(n_blobs_crossings * ratio_validation)

    # split training and validation
    validation_blobs = {
        "individuals": individuals[:n_individual_blobs_validation],
        "crossings": crossings[:n_crossing_blobs_validation],
    }

    training_blobs = {
        "individuals": individuals[n_individual_blobs_validation:],
        "crossings": crossings[n_crossing_blobs_validation:],
    }

    ratio_crossings = n_blobs_crossings / (n_blobs_crossings + n_blobs_individuals)
    training_blobs["weights"] = [ratio_crossings, 1 - ratio_crossings]

    logging.info(
        f"{len(training_blobs['individuals'])} individual and "
        f"{len(training_blobs['crossings'])} crossing blobs for training\n"
        f"{len(validation_blobs['individuals'])} individual and "
        f"{len(validation_blobs['crossings'])} crossing blobs for validation\n"
        f"{len(toassign_blobs)} blobs to test"
    )

    return training_blobs, validation_blobs, toassign_blobs
