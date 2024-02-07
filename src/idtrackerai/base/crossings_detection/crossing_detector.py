import logging

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import ListOfBlobs, Session
from idtrackerai.utils import conf, load_id_images

from ..network import (
    CNN,
    DEVICE,
    LearnerClassification,
    NetworkParams,
    StopTraining,
    get_dataloader,
    get_predictions,
    train_loop,
)
from .crossings_dataset import get_train_validation_and_eval_blobs
from .model_area import ModelArea


def apply_area_and_unicity_heuristics(list_of_blobs: ListOfBlobs, n_animals: int):
    logging.info(
        "Classifying Blobs as individuals or crossings "
        "depending on their area and the number of blobs in the frame"
    )

    model_area = ModelArea(list_of_blobs, n_animals)

    for blobs_in_frame in list_of_blobs.blobs_in_video:
        unicity_cond = len(blobs_in_frame) == n_animals
        for blob in blobs_in_frame:
            blob.seems_like_individual = unicity_cond or model_area(blob.area)


def detect_crossings(list_of_blobs: ListOfBlobs, session: Session):
    """Classify all blobs in the video as being crossings or individuals"""

    apply_area_and_unicity_heuristics(list_of_blobs, session.n_animals)

    train_images, train_labels, train_weights, val_images, val_labels = (
        get_train_validation_and_eval_blobs(
            list_of_blobs.blobs_in_video, session.n_animals
        )
    )

    unknown_blobs = [
        blob
        for blob in list_of_blobs.all_blobs
        if not hasattr(blob, "is_an_individual")
    ]

    logging.info(f"{len(unknown_blobs)} unknown blobs")

    if (
        np.count_nonzero(train_labels)
        < conf.MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR
    ):
        logging.debug("There are not enough crossings to train the crossing detector")
        for blob in unknown_blobs:
            blob.is_an_individual = blob.seems_like_individual
        return
    logging.info("There are enough crossings to train the crossing detector")

    train_loader = get_dataloader(
        "training",
        load_id_images(session.id_images_file_paths, train_images),
        train_labels,
        conf.BATCH_SIZE_DCD,
    )

    val_loader = get_dataloader(
        "validation",
        load_id_images(session.id_images_file_paths, val_images),
        val_labels,
    )

    logging.info("Setting crossing detector network parameters")
    network_params = NetworkParams(
        n_classes=2,
        architecture="CNN",
        save_folder=session.crossings_detector_folder,
        model_name="crossing_detector",
        image_size=session.id_image_size,
        optimizer="Adam",
        schedule=[30, 60],
        optim_args={"lr": conf.LEARNING_RATE_DCD},
        epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_DCD,
    )
    network_params.save()

    crossing_model = CNN.from_network_params(network_params).to(DEVICE)

    if network_params.optimizer == "Adam":
        optimizer = Adam(crossing_model.parameters(), **network_params.optim_args)
    elif network_params.optimizer == "SGD":
        optimizer = SGD(crossing_model.parameters(), **network_params.optim_args)
    else:
        raise AttributeError(network_params.optimizer)

    scheduler = MultiStepLR(optimizer, milestones=network_params.schedule, gamma=0.1)
    criterion = CrossEntropyLoss(
        weight=torch.tensor(train_weights, dtype=torch.float32)
    ).to(DEVICE)
    learner = LearnerClassification(crossing_model, criterion, optimizer, scheduler)
    stopping = StopTraining(
        epochs_limit=conf.MAXIMUM_NUMBER_OF_EPOCHS_DCD,
        overfitting_limit=conf.OVERFITTING_COUNTER_THRESHOLD_DCD,
        plateau_limit=conf.LEARNING_RATIO_DIFFERENCE_DCD,
    )

    try:
        train_loop(learner, train_loader, val_loader, stopping)
    except RuntimeError as exc:
        logging.warning(
            "[red]The model diverged[/] provably due to a bad segmentation. Falling"
            " back to individual-crossing discrimination by average area model."
            " Original error: %s",
            exc,
            extra={"markup": True},
        )
        for blob in unknown_blobs:
            blob.is_an_individual = blob.seems_like_individual
        return

    del train_loader
    del val_loader

    learner.save_model(network_params.model_path)
    logging.info("Using crossing detector to classify individuals and crossings")
    predictions, _softmax = get_predictions(
        crossing_model,
        [(blob.id_image_index, blob.episode) for blob in unknown_blobs],
        session.id_images_file_paths,
        "crossings",
    )

    logging.info(
        "Prediction results: %d individuals and %d crossings",
        np.count_nonzero(predictions == 1),
        np.count_nonzero(predictions == 2),
    )
    for blob, prediction in zip(unknown_blobs, predictions):
        blob.is_an_individual = prediction != 2

    list_of_blobs.update_id_image_dataset_with_crossings(session.id_images_file_paths)
