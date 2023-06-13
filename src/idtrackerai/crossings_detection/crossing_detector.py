import logging

import torch
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import ListOfBlobs, Video
from idtrackerai.network import (
    LearnerClassification,
    NetworkParams,
    get_device,
    weights_xavier_init,
)
from idtrackerai.utils import conf

from .crossings_dataset import (
    get_train_validation_and_eval_blobs,
    get_training_data_loaders,
)
from .crossings_network import (
    StopTraining,
    get_predictions_crossigns,
    train_deep_crossing,
)
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


def detect_crossings(list_of_blobs: ListOfBlobs, video: Video):
    """Classify all blobs in the video as being crossings or individuals"""

    apply_area_and_unicity_heuristics(list_of_blobs, video.number_of_animals)

    train_blobs, val_blobs, eval_blobs = get_train_validation_and_eval_blobs(
        list_of_blobs.blobs_in_video, video.number_of_animals
    )

    if (
        len(train_blobs["crossings"])
        < conf.MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR
    ):
        logging.debug("There are not enough crossings to train the crossing detector")
        for blob in eval_blobs:
            blob.is_an_individual = blob.seems_like_individual
        return
    logging.info("There are enough crossings to train the crossing detector")
    train_loader, val_loader = get_training_data_loaders(
        video.id_images_file_paths, train_blobs, val_blobs
    )
    logging.info("Setting crossing detector network parameters")
    network_params = NetworkParams(
        number_of_classes=2,
        architecture="DCD",
        save_folder=video.crossings_detector_folder,
        model_name="crossing_detector",
        image_size=video.id_image_size,
        optimizer="Adam",
        schedule=[30, 60],
        optim_args={"lr": conf.LEARNING_RATE_DCD},
        epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_DCD,
    )
    network_params.save()
    logging.info("Setting training criterion")
    criterion = CrossEntropyLoss(weight=torch.tensor(train_blobs["weights"]))
    crossing_detector_model = LearnerClassification.create_model(network_params)
    logging.info("Initialize networks params with Xavier initialization")
    crossing_detector_model.apply(weights_xavier_init)

    logging.info("Sending model and criterion to GPU")
    cudnn.benchmark = True  # make it train faster
    crossing_detector_model = crossing_detector_model.to(get_device())
    criterion = criterion.to(get_device())

    logging.info(f"Setting {network_params.optimizer} optimizer")
    if network_params.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            crossing_detector_model.parameters(), **network_params.optim_args
        )
    elif network_params.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            crossing_detector_model.parameters(), **network_params.optim_args
        )
    else:
        raise AttributeError(network_params.optimizer)

    logging.info("Setting scheduler")
    scheduler = MultiStepLR(optimizer, milestones=network_params.schedule, gamma=0.1)

    learner = LearnerClassification(
        crossing_detector_model, criterion, optimizer, scheduler
    )

    stop_training = StopTraining(network_params.epochs)

    model_diverged, best_model_path = train_deep_crossing(
        learner, train_loader, val_loader, network_params, stop_training
    )

    if model_diverged:
        logging.warning(
            (
                "[red]The model diverged[/] provably due to a bad segmentation. Falling"
                " back to individual-crossing discrimination by average area model."
            ),
            extra={"markup": True},
        )
        for blob in eval_blobs:
            blob.is_an_individual = blob.seems_like_individual
        return

    del train_loader
    del val_loader

    model_state = torch.load(best_model_path)
    crossing_detector_model.load_state_dict(model_state, strict=True)
    logging.info("Loaded best model weights from %s", best_model_path)

    logging.info("Using crossing detector to classify individuals and crossings")
    predictions = get_predictions_crossigns(
        video.id_images_file_paths, crossing_detector_model, eval_blobs
    )

    logging.info(
        "Prediction results: %d individuals and %d crossings",
        predictions.count(0),
        predictions.count(1),
    )
    for blob, prediction in zip(eval_blobs, predictions):
        blob.is_an_individual = prediction != 1

    list_of_blobs.update_id_image_dataset_with_crossings(video.id_images_file_paths)
