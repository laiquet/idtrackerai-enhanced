import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import ListOfBlobs, Session, __version__, conf
from idtrackerai.utils import create_dir, load_id_images

from ..network import (
    DEVICE,
    IdCNN,
    IdentifierIdCNN,
    StopTraining,
    get_dataloader,
    get_predictions,
    train_loop,
)
from .crossings_dataset import get_train_validation_and_eval_blobs
from .model_area import ModelArea


def _apply_area_and_unicity_heuristics(
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


def _update_id_image_dataset_with_crossings(
    list_of_blobs: ListOfBlobs, id_images_file_paths: list[Path]
):
    """Adds a array to the identification images files indicating whether
    each image is an individual or a crossing.
    """
    logging.info("Updating crossings in identification images files")

    crossings = []
    for path in id_images_file_paths:
        with h5py.File(path, "r") as file:
            crossings.append(np.empty(len(file["id_images"]), bool))  # type: ignore

    for blob in list_of_blobs.all_blobs:
        crossings[blob.episode][blob.id_image_index] = blob.is_a_crossing

    for path, crossing in zip(id_images_file_paths, crossings):
        try:
            with h5py.File(path, "r+") as file:
                if "crossings" in file:  # in case we are reusing the file in API calls
                    logging.warning("Overwriting existing crossings in %s", path)
                    del file["crossings"]
                file.create_dataset("crossings", data=crossing)
        except BlockingIOError as exc:
            # Some MacOS crash with
            # BlockingIOError: [Errno 35] Unable to open file (unable to lock file, errno = 35, error message = 'Resource temporarily unavailable')
            logging.error(f"Failed at writing in {path}: {exc}")


def detect_crossings(list_of_blobs: ListOfBlobs, session: Session) -> None:
    """Classify all blobs in the video as being crossings or individuals"""

    _apply_area_and_unicity_heuristics(list_of_blobs, session.n_animals)

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
        np.bincount(train_labels, minlength=2).min()
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

    crossing_model = IdCNN(input_shape=session.id_image_size, out_dim=2).to(DEVICE)
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
    if conf.TORCH_COMPILE:
        crossing_model.compile()
        criterion.compile()

    with (session.crossings_detector_folder / "model_params.json").open("w") as file:
        json.dump(
            {
                "n_classes": 2,
                "image_size": session.id_image_size,
                "resolution_reduction": session.resolution_reduction,
                "model": crossing_model.__class__.__name__,
                "version": __version__,
            },
            file,
            indent=4,
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
        IdentifierIdCNN(crossing_model),
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

    _update_id_image_dataset_with_crossings(list_of_blobs, session.id_images_file_paths)
