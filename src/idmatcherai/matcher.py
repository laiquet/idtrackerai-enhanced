import json
import logging
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from idtrackerai.network import LearnerClassification, NetworkParams
from idtrackerai.tracker.identity_network import get_predictions_identities


def match(id_images_path: Path, model_path: Path):
    logging.info(
        "Matching images from %s with model from %s", id_images_path, model_path
    )

    id_images_paths = list(id_images_path.glob("id_images_*.hdf5"))

    labels_for_episode = extract_labels_per_episode(id_images_paths)

    set_of_labels = set(np.concatenate(labels_for_episode))
    set_of_labels.discard(0)

    n_img_ids = len(set_of_labels)
    """number of labels in the images to be assigned by the model"""

    model, model_params = load_identification_model(model_path)
    n_model_ids = model_params.number_of_classes
    """number of classes in the model"""

    matching = np.zeros((n_img_ids, n_model_ids), int)

    for identity in set_of_labels:
        images = extact_images_for_id(id_images_paths, labels_for_episode, identity)
        predictions, softmax_probs = get_predictions_identities(
            model, images, model_params
        )

        matching[identity - 1] = np.bincount(predictions, minlength=n_model_ids + 1)[1:]
    return matching


def extact_images_for_id(
    id_images_file_paths: Iterable[Path],
    labels_per_episode: list[np.ndarray],
    identity: int,
) -> np.ndarray:
    images = []

    for labels, path in zip(labels_per_episode, id_images_file_paths):
        selected_indices = labels == identity
        if not any(selected_indices):
            continue

        with h5py.File(path, "r") as file:
            images.append(
                file["id_images"][selected_indices]  # type: ignore
                if "id_images" in file
                else file["identification_images"][selected_indices]  # type: ignore
            )
    images = np.concatenate(images)
    logging.info("Extracting %d images for identity %d", len(images), identity)
    return images


def extract_labels_per_episode(id_images_file_paths: Iterable[Path]):
    labels = []
    for path in id_images_file_paths:
        with h5py.File(path, "r") as file:
            identities: np.ndarray = file["identities"][:]  # type: ignore

            # v4 compatibility
            if identities.ndim == 2:
                identities = np.squeeze(identities)
            if not issubclass(identities.dtype.type, np.integer):
                identities[np.isnan(identities)] = 0
                identities = identities.astype(int)

            # prerelease compatibility
            identities[identities < 0] = 0

            labels.append(identities)

    return labels


def load_identification_model(model_folder: Path):
    params_path = model_folder / "model_params.json"
    if params_path.is_file():
        with open(params_path, "rb") as file:
            params = json.load(file)
    elif params_path.with_suffix(".npy").is_file():
        params = np.load(params_path.with_suffix(".npy"), allow_pickle=True).item()
    else:
        raise FileNotFoundError(params_path)

    identification_network_params = NetworkParams(
        schedule=params["schedule"],
        number_of_classes=params["number_of_classes"],
        architecture="idCNN",
        restore_folder=model_folder,
        model_name=params["model_name"],
        image_size=params["image_size"],
    )

    identification_model = LearnerClassification.load_model(
        identification_network_params
    )
    return identification_model, identification_network_params
