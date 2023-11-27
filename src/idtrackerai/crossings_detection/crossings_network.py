import logging
import sys
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.status import Status

from idtrackerai import Blob
from idtrackerai.network import (
    CNN,
    DEVICE,
    DataLoaderWithLabels,
    LearnerClassification,
    NetworkParams,
    evaluate,
    train,
)
from idtrackerai.utils import conf, track

from .crossings_dataset import get_crossing_dataloader


class StopTraining:
    """CROSSING Stops the training of the network according to the conditions specified
    in __call__
    """

    epochs_before_checking_stopping_conditions = 10

    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs  # maximal num of epochs
        self.overfitting_counter: int = 0
        """Number of epochs in which the network is overfitting before
        stopping the training"""

        self.epochs_completed = -1

    def __call__(
        self,
        loss_training: float,
        loss_validation: list[float],
        accuracy_validation: float,
        status: Status,
    ):
        """Returns True when one of the conditions to stop the training is
        satisfied, otherwise it returns False"""
        self.epochs_completed += 1

        if self.epochs_completed > 0 and (
            np.isnan(loss_training) or np.isnan(loss_validation[-1])
        ):
            status.stop()
            logging.error(
                "The model diverged with loss NaN, falling back "
                "to detecting crossings with the model area"
            )
            return True
        # check if it did not reached the epochs limit
        if self.epochs_completed >= self.num_epochs:
            status.stop()
            logging.warning(
                "The number of epochs completed is larger than the number "
                "of epochs set for training, we stop the training"
            )
            return True

        if self.epochs_completed <= self.epochs_before_checking_stopping_conditions:
            return False

        # check that the model is not overfitting or if it reached
        # a stable saddle (minimum)
        current_loss = loss_validation[-1]
        previous_loss = np.nanmean(
            loss_validation[-self.epochs_before_checking_stopping_conditions : -1]
        )

        # The validation loss in the first 10 epochs could have exploded
        # but being decreasing.
        if np.isnan(previous_loss):
            previous_loss = sys.float_info[0]
        losses_difference = float(previous_loss) - current_loss

        # check overfitting
        if losses_difference < 0.0:
            self.overfitting_counter += 1
            if self.overfitting_counter >= conf.OVERFITTING_COUNTER_THRESHOLD_DCD:
                status.stop()
                logging.info("Overfitting")
                return True
        else:
            self.overfitting_counter = 0
        # check if the error is not decreasing much
        if abs(losses_difference) < conf.LEARNING_RATIO_DIFFERENCE_DCD * current_loss:
            status.stop()
            logging.info("The losses difference is very small, we stop the training")
            return True
        # if the individual accuracies in validation are 1. for all the animals
        if accuracy_validation == 1.0:
            status.stop()
            logging.info("The accuracy in validation is 100%, we stop the training")
            return True
        # if the validation loss is 0.
        if previous_loss == 0.0 or current_loss == 0.0:
            status.stop()
            logging.info("The validation loss is 0.0, we stop the training")
            return True

        return False


def train_deep_crossing(
    learner: LearnerClassification,
    train_loader: DataLoaderWithLabels,
    val_loader: DataLoaderWithLabels,
    network_params: NetworkParams,
    stop_training: StopTraining,
) -> tuple[bool, Path]:
    logging.info("Training Deep Crossing Detector")

    # Initialize metric storage
    train_loss = 0.0
    val_loss = 0.0
    val_losses = []
    val_acc = 0.0

    logging.debug("Entering the epochs loop...")
    with Console().status("[red]Epochs loop...") as status:
        while not stop_training(train_loss, val_losses, val_acc, status):
            epoch = stop_training.epochs_completed

            train_loss = train(epoch, train_loader, learner)
            val_loss, val_acc = evaluate(val_loader, learner)

            val_losses.append(val_loss)

            with suppress(IndexError):
                status.update(
                    f"[red]Epoch {epoch}: training loss ="
                    f" {train_loss:.5f}, validation loss ="
                    f" {val_loss:.5f} and accuracy = {val_acc:.3%}"
                )

        logging.info("Last epoch: %s", status.status, extra={"markup": True})

    learner.save_model(network_params.model_path)
    return np.isnan(train_loss) or np.isnan(val_loss), network_params.model_path


def get_predictions_crossigns(
    id_images_file_paths: list[Path], model: CNN, blobs: list[Blob]
):
    loader = get_crossing_dataloader(id_images_file_paths, blobs, "test")

    model.to(DEVICE)
    model.eval()
    predictions = np.empty(len(blobs), np.int32)
    index = 0
    with torch.no_grad():
        for input, _target in track(loader, "Predicting crossings"):
            # Inference
            output = model.forward(input.to(DEVICE))
            # https://github.com/pytorch/pytorch/issues/92311
            pred = output.max(dim=1).indices

            predictions[index : index + len(pred)] = pred.cpu()
            index += len(pred)
    assert index == len(predictions)
    return predictions
