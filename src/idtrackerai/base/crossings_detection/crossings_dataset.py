import logging

import numpy as np

from idtrackerai import Blob
from idtrackerai.utils import conf, track


def get_train_validation_and_eval_blobs(
    blobs_in_video: list[list[Blob]], number_of_animals: int, ratio_val: float = 0.1
):
    """Given a list of blobs return 2 dictionaries (training_blobs, validation_blobs),
    and a list (toassign_blobs).

    :param list_of_blobs:
    :param ratio_validation:
    :return: training_blobs, validation_blobs, toassign_blobs
    """
    logging.info("Get list of blobs for training, validation and eval")

    individuals = []
    crossings = []
    for blobs_in_frame in track(blobs_in_video, "First individual/crossing assignment"):
        in_a_global_fragment_core = len(blobs_in_frame) == number_of_animals
        for blob in blobs_in_frame:
            if in_a_global_fragment_core or blob.is_a_sure_individual():
                blob.used_for_training_crossings = True
                blob.is_an_individual = True
                individuals.append((blob.id_image_index, blob.episode))
            elif blob.is_a_sure_crossing():
                blob.used_for_training_crossings = True
                blob.is_an_individual = False
                crossings.append((blob.id_image_index, blob.episode))
            # unknown blobs are recognized by not having the attribute `is_an_individual`

    individuals = np.asarray(individuals) if individuals else np.empty((0, 2))
    crossings = np.asarray(crossings) if crossings else np.empty((0, 2))

    # clear no longer useful cached properties
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            blob.__dict__.pop("has_a_next_crossing", None)
            blob.__dict__.pop("has_a_previous_crossing", None)
            blob.__dict__.pop("has_multiple_next", None)
            blob.__dict__.pop("has_multiple_previous", None)

    logging.debug(
        f"{len(individuals)} sure individuals and {len(crossings)} sure crossings"
    )

    # Shuffle
    rng = np.random.default_rng()
    rng.shuffle(individuals)
    rng.shuffle(crossings)

    # Limit the number of images
    crossings = crossings[: conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR]
    individuals = individuals[: conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR]

    # Split train and val
    n_blobs_crossings = len(crossings)
    n_blobs_individuals = len(individuals)
    n_individual_blobs_val = int(n_blobs_individuals * ratio_val)
    n_crossing_blobs_val = int(n_blobs_crossings * ratio_val)

    val_individual = individuals[:n_individual_blobs_val]
    val_crossing = crossings[:n_crossing_blobs_val]
    train_individual = individuals[n_individual_blobs_val:]
    train_crossing = crossings[n_crossing_blobs_val:]

    val_images = np.concatenate((val_individual, val_crossing))
    val_labels = np.concatenate(
        (np.zeros(len(val_individual), np.int64), np.ones(len(val_crossing), np.int64))
    )

    train_images = np.concatenate((train_individual, train_crossing))
    train_labels = np.concatenate(
        (
            np.zeros(len(train_individual), np.int64),
            np.ones(len(train_crossing), np.int64),
        )
    )

    ratio_crossings = n_blobs_crossings / (n_blobs_crossings + n_blobs_individuals)
    train_weights = [ratio_crossings, 1 - ratio_crossings]

    logging.info(
        f"{len(train_individual)} individual and "
        f"{len(train_crossing)} crossing blobs for training\n"
        f"{len(val_individual)} individual and "
        f"{len(val_crossing)} crossing blobs for validation"
    )

    return train_images, train_labels, train_weights, val_images, val_labels
