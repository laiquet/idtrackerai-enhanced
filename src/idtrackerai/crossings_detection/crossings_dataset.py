import logging
import platform
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from idtrackerai import Blob
from idtrackerai.network import DataLoaderWithLabels, ImageDataset
from idtrackerai.tracker.identity_dataset import duplicate_PCA_images
from idtrackerai.utils import conf, load_id_images, track


def get_train_validation_and_eval_blobs(
    blobs_in_video: list[list[Blob]],
    number_of_animals: int,
    ratio_validation: float = 0.1,
) -> tuple[dict[str, list[Blob]], dict[str, list[Blob]], list[Blob]]:
    """Given a list of blobs return 2 dictionaries (training_blobs, validation_blobs),
    and a list (toassign_blobs).

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
                blob.is_an_individual = True
                individuals.append(blob)
            elif blob.is_a_sure_crossing():
                blob.used_for_training_crossings = True
                blob.is_an_individual = False
                crossings.append(blob)
            else:
                blob.used_for_training_crossings = False
                toassign_blobs.append(blob)

    # clear no longer useful cached properties
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            blob.__dict__.pop("has_a_next_crossing", None)
            blob.__dict__.pop("has_a_previous_crossing", None)
            blob.__dict__.pop("has_multiple_next", None)
            blob.__dict__.pop("has_multiple_previous", None)

    logging.debug(
        f"{len(individuals)} individual, "
        f"{len(crossings)} crossing and "
        f"{len(toassign_blobs)} unknown blobs in total"
    )

    # Shuffle and make crossings and individuals even
    rng = np.random.default_rng()
    rng.shuffle(individuals)
    rng.shuffle(crossings)

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


if platform.system() in ("Windows", "Darwin"):
    # Using multiprocessing in Windows and MacOS causes a
    # recursion limit error difficult to debug
    num_workers = 0
else:
    num_workers = 1


def get_crossing_dataloader(
    id_images_file_paths: list[Path],
    blobs: list[Blob] | dict[str, list[Blob]],
    scope: Literal["training", "validation", "test"],
) -> DataLoaderWithLabels:
    logging.info("Creating %s CrossingDataset", scope)

    if isinstance(blobs, dict):
        images = [
            (blob.id_image_index, blob.episode) for blob in blobs["crossings"]
        ] + [(blob.id_image_index, blob.episode) for blob in blobs["individuals"]]

        images = load_id_images(id_images_file_paths, images)
        images = np.expand_dims(images, axis=-1)

        labels = np.concatenate(
            [np.ones(len(blobs["crossings"])), np.zeros(len(blobs["individuals"]))],
            axis=0,
        )

        if scope == "training":
            images, labels = duplicate_PCA_images(images, labels)

    elif isinstance(blobs, list):
        images = load_id_images(
            id_images_file_paths,
            [(blob.id_image_index, blob.episode) for blob in blobs],
        )
        images = np.expand_dims(images, axis=-1)
        labels = np.zeros(len(images))

    dataset = ImageDataset(images, labels, transform=transforms.ToTensor())

    return DataLoader(
        dataset,
        batch_size=(
            conf.BATCH_SIZE_DCD
            if scope == "training"
            else conf.BATCH_SIZE_PREDICTIONS_DCD
        ),
        shuffle=scope == "training",
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
