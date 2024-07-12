import json
import logging

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import ListOfBlobs, Session
from idtrackerai.utils import conf, create_dir, load_id_images

from ..network import (
    CNN,
    DEVICE,
    IdentifierCNN,
    StopTraining,
    get_dataloader,
    get_predictions,
    train_loop,
)
from .crossings_dataset import get_train_validation_and_eval_blobs
from .model_area import ModelArea


def apply_area_and_unicity_heuristics(
    list_of_blobs: ListOfBlobs, n_animals: int
) -> None:
    logging.info(
        "Classifying Blobs as individuals or crossings "
        "depending on their area and the number of blobs in the frame"
    )

    model_area = ModelArea(list_of_blobs, n_animals)

    for blobs_in_frame in list_of_blobs.blobs_in_video:
        unicity_cond = len(blobs_in_frame) == n_animals
        for blob in blobs_in_frame:
            blob.seems_like_individual = unicity_cond or model_area(blob.area)

    n_seems_like_individual = sum(
        blob.seems_like_individual for blob in list_of_blobs.all_blobs
    )
    logging.info(
        f"{n_seems_like_individual} blobs seem like individuals, "
        f"{list_of_blobs.number_of_blobs - n_seems_like_individual} seem like crossings"
    )


def detect_crossings(list_of_blobs: ListOfBlobs, session: Session) -> None:
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

    create_dir(session.crossings_detector_folder, remove_existing=True)

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

    with (session.crossings_detector_folder / "model_params.json").open("w") as file:
        json.dump({"n_classes": 2, "image_size": session.id_image_size}, file)

    crossing_model = CNN(input_shape=session.id_image_size, out_dim=2).to(DEVICE)
    optimizer = Adam(crossing_model.parameters(), lr=conf.LEARNING_RATE_DCD)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    criterion = CrossEntropyLoss(torch.tensor(train_weights, dtype=torch.float32)).to(
        DEVICE
    )
    stopping = StopTraining(
        epochs_limit=conf.MAXIMUM_NUMBER_OF_EPOCHS_DCD,
        overfitting_limit=conf.OVERFITTING_COUNTER_THRESHOLD_DCD,
        plateau_limit=conf.LEARNING_RATIO_DIFFERENCE_DCD,
    )

    try:
        train_loop(
            crossing_model,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            stopping,
            scheduler,
        )
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

    model_path = session.crossings_detector_folder / "crossing_detector.model.pt"
    logging.info("Saving model at %s", model_path)
    torch.save(crossing_model.state_dict(), model_path)

    logging.info("Using crossing detector to classify individuals and crossings")
    predictions, _softmax = get_predictions(
        IdentifierCNN(crossing_model),
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
