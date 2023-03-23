import json
import logging
from pathlib import Path

import numpy as np

from idtrackerai import Fragment
from idtrackerai.network import LearnerClassification, NetworkParams
from idtrackerai.tracker.network.get_predictions import GetPredictionsIdentities

from .images import extact_all_images_and_labels


def load_identification_model(model_folder: Path):
    params_path = model_folder / "model_params.json"
    if params_path.is_file():
        with open(params_path, "rb") as file:
            params: dict = json.load(file)
    elif params_path.with_suffix(".npy").is_file():
        params: dict = np.load(
            params_path.with_suffix(".npy"), allow_pickle=True
        ).item()
    else:
        raise FileNotFoundError(params_path)

    identification_network_params = NetworkParams(
        schedule=params["schedule"],
        number_of_classes=params["number_of_classes"],
        architecture="idCNN",
        restore_folder=model_folder,
        model_name=params["model_name"],
        dataset=params["dataset"],
        saveid=params["saveid"],
        image_size=params["image_size"],
        use_gpu=True,
    )

    # Initialize network
    logging.info("Creating model")
    identification_model = LearnerClassification.load_model(
        identification_network_params
    )
    return identification_model, identification_network_params


def match(id_images_path: Path, model_path: Path):
    logging.info(
        "Matching images from %s with model from %s", id_images_path, model_path
    )

    images, labels = extact_all_images_and_labels(
        id_images_path.glob("id_images_*.hdf5")
    )
    model, model_params = load_identification_model(model_path)

    set_of_labels = set(labels.astype(int))
    set_of_labels.discard(0)

    num_labels = len(set_of_labels)
    """number of labels in the images to be assigned by the model (B)"""
    num_classes = model_params.number_of_classes
    """number of classes in the model (A)"""
    # TODO if num_labels <= num_classes:

    confusion_matrix = np.zeros((num_classes, num_labels))
    frequencies_matrix = np.zeros((num_classes, num_labels), int)

    for identity in set_of_labels:
        assigner = GetPredictionsIdentities(
            model, images[labels == identity], model_params
        )
        assigner.get_all_predictions()
        frequencies, P1_vector = compute_identification_statistics(assigner)
        confusion_matrix[identity - 1] = P1_vector
        frequencies_matrix[identity - 1] = frequencies
    return confusion_matrix, frequencies_matrix


def compute_identification_statistics(assigner: GetPredictionsIdentities):
    frequencies = Fragment.compute_identification_frequencies_individual_fragment(
        assigner._predictions, assigner.network_params.number_of_classes
    )
    P1_vector = Fragment.compute_P1_from_frequencies(frequencies)

    return frequencies, P1_vector
