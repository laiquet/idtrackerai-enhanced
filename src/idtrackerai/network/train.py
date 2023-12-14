import logging
import sys
from itertools import count
from typing import Callable

import numpy as np
import torch
from rich.console import Console

from . import CNN, DEVICE, DataLoaderWithLabels, LearnerClassification


class StopTraining:
    epochs_before_checking_stopping_conditions: int
    overfitting_counter: int
    """Number of epochs in which the network is overfitting before
    stopping the training"""

    loss_history: list[float] = []
    is_first_accumulation: bool
    epochs_limit: int
    overfitting_limit: int
    plateau_limit: float

    def __init__(
        self,
        epochs_limit: int,
        overfitting_limit: int,
        plateau_limit: float,
        is_first_accumulation: bool = False,
    ):
        self.epochs_before_checking_stopping_conditions = 10
        self.overfitting_counter = 0
        self.loss_history: list[float] = []
        self.is_first_accumulation: bool = is_first_accumulation
        self.epochs_limit = epochs_limit
        self.overfitting_limit = overfitting_limit
        self.plateau_limit = plateau_limit

    def __call__(self, train_loss: float, val_loss: float, val_acc: float) -> str:
        self.loss_history.append(val_loss)

        if self.epochs_completed > 1 and (np.isnan(train_loss) or np.isnan(val_loss)):
            raise RuntimeError(
                f"The model diverged {train_loss=} {val_loss=}. Check the"
                " hyperparameters and the architecture of the network."
            )

        # check if it did not reached the epochs limit
        if self.epochs_completed >= self.epochs_limit:
            return (
                "The number of epochs completed is larger than the number "
                "of epochs set for training, we stop the training"
            )

        if self.epochs_completed <= self.epochs_before_checking_stopping_conditions:
            return ""

        # check that the model is not overfitting or if it reached
        # a stable saddle (minimum)
        loss_trend = np.nanmean(
            self.loss_history[-self.epochs_before_checking_stopping_conditions : -1]
        )

        # The validation loss in the first 10 epochs could have exploded
        # but being decreasing.
        if np.isnan(loss_trend):
            loss_trend = sys.float_info[0]
        losses_difference = float(loss_trend) - val_loss

        # check overfitting
        if losses_difference < 0.0:
            self.overfitting_counter += 1
            if self.overfitting_counter >= self.overfitting_limit:
                return "Overfitting"
        else:
            self.overfitting_counter = 0

        # check if the error is not decreasing much

        if abs(losses_difference) < self.plateau_limit * val_loss:
            return "The losses difference is very small, we stop the training"

        # if the individual accuracies in validation are 1. for all the animals
        if val_acc == 1.0:
            return (
                "The individual accuracies in validation is 100%, we stop the training"
            )

        # if the validation loss is 0.
        if loss_trend == 0.0 or val_loss == 0.0:
            return "The validation loss is 0.0, we stop the training"

        return ""

    @property
    def epochs_completed(self):
        return len(self.loss_history)


def train_loop(
    learner: LearnerClassification,
    train_loader: DataLoaderWithLabels,
    val_loader: DataLoaderWithLabels,
    stop_training: Callable[[float, float, float], str],
):
    logging.debug("Entering the training loop...")
    with Console().status("[red]Epochs loop...") as status:
        for epoch in count(1):
            train_loss = train(train_loader, learner)
            val_loss, val_acc = evaluate(val_loader, learner)

            status.update(
                f"[red]Epoch {epoch}: training loss = {train_loss:.5f}, validation loss"
                f" = {val_loss:.5f} and accuracy = {val_acc:.3%}"
            )
            stop_message = stop_training(train_loss, val_loss, val_acc)
            if stop_message:
                break
        else:
            raise

    logging.info(stop_message)
    logging.info("Last epoch: %s", status.status, extra={"markup": True})
    logging.info("Network trained")


def train(train_loader: DataLoaderWithLabels, learner: LearnerClassification):
    """Trains trains a network using a learner, a given train_loader"""
    losses = 0
    n_predictions = 0

    learner.train()

    for input, target in train_loader:
        loss = learner.learn(input.to(DEVICE), target.to(DEVICE))

        losses += loss.item() * len(input)
        n_predictions += len(input)

    learner.step_schedule()
    return losses / n_predictions


def evaluate(eval_loader: DataLoaderWithLabels, learner: LearnerClassification):
    with torch.no_grad():
        losses = 0
        n_predictions = 0
        n_right_guess = 0

        learner.eval()

        for input, target in eval_loader:
            target = target.to(DEVICE)

            loss, output = learner.forward_with_criterion(input.to(DEVICE), target)
            n_predictions += len(target)
            n_right_guess += (output.max(1).indices == target).count_nonzero().item()

            losses += loss.item() * len(input)

    return losses / n_predictions, n_right_guess / n_predictions


def evaluate_only_acc(eval_loader: DataLoaderWithLabels, model: CNN):
    with torch.no_grad():
        model.eval()
        n_predictions = 0
        n_right_guess = 0

        for input, target in eval_loader:
            predictions = model.forward(input.to(DEVICE)).max(1).indices
            n_predictions += len(target)
            n_right_guess += (predictions == target.to(DEVICE)).count_nonzero().item()

    return n_right_guess / n_predictions
