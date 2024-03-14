import logging
import os
from functools import partial
from itertools import count
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from rich.console import Console
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import VisionDataset

from idtrackerai.utils import conf, load_id_images, track

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
            loss_trend = float("inf")
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
                f"[red]Epoch {epoch:2}: training loss = {train_loss:.5f}, validation"
                f" loss = {val_loss:.5f} and accuracy = {val_acc:.3%}"
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

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        loss = learner.learn(images, labels)
        losses += loss.item() * len(images)
        n_predictions += len(images)

    learner.step_schedule()
    return losses / n_predictions


@torch.inference_mode()
def evaluate(eval_loader: DataLoaderWithLabels, learner: LearnerClassification):
    losses = 0
    n_predictions = 0
    n_right_guess = 0

    learner.eval()

    for images, labels in eval_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        loss, output = learner.forward_with_criterion(images, labels)
        n_predictions += len(labels)
        n_right_guess += (output.max(1).indices == labels).count_nonzero().item()

        losses += loss.item() * len(images)

    return losses / n_predictions, n_right_guess / n_predictions


@torch.inference_mode()
def evaluate_only_acc(eval_loader: DataLoaderWithLabels, model: CNN):
    model.eval()
    n_predictions = 0
    n_right_guess = 0

    for images, labels in eval_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        predictions = model.forward(images).max(1).indices
        n_predictions += len(labels)
        n_right_guess += (predictions == labels).count_nonzero().item()

    return n_right_guess / n_predictions


class ImageDataset(VisionDataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray | None, transform=None):
        super().__init__("", transform=transform)

        if images.ndim <= 3:
            images = np.expand_dims(images, axis=-1)

        self.images = images
        self.labels = (
            labels.astype(np.int64)
            if labels is not None
            else np.zeros(len(images), np.int64)
        )
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_dataloader(
    scope: Literal["training", "validation", "test"],
    images: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = conf.BATCH_SIZE_PREDICTIONS,
) -> DataLoaderWithLabels:
    logging.info(
        "Creating %s dataloader with %d images"
        + (
            f" labeled with {len(np.unique(labels))} distinct classes"
            if labels is not None
            else ""
        ),
        scope,
        len(images),
    )

    if scope == "training":
        assert labels is not None
        rotated_images = np.rot90(images, 2, axes=(1, 2))
        images = np.concatenate([images, rotated_images])
        labels = np.concatenate([labels, labels])

    return DataLoader(
        ImageDataset(images, labels, transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=scope == "training",
        num_workers=1 if os.name == "nt" else 4,  # windows
        persistent_workers=True,
        pin_memory=True,
    )


@torch.inference_mode()
def get_predictions(
    model: CNN,
    image_location: Sequence[tuple[int, int]] | np.ndarray,
    id_images_paths: list[Path],
    kind: str = "identities",
):
    logging.debug("Predicting %s of %d images", kind, len(image_location), stacklevel=3)
    predictions = np.empty(len(image_location), np.int32)
    max_softmax = np.empty(len(image_location), np.float32)
    index = 0
    model.eval()
    dataloader = get_onthefly_dataloader(image_location, id_images_paths)
    for images, _labels in track(dataloader, "Predicting " + kind):
        softmax = functional.softmax(model.forward(images.to(DEVICE)), dim=1)
        # https://github.com/pytorch/pytorch/issues/92311
        maximum, pred = softmax.max(dim=1)

        predictions[index : index + len(pred)] = (pred + 1).cpu()
        max_softmax[index : index + len(pred)] = maximum.cpu()
        index += len(pred)
    assert index == len(predictions) == len(max_softmax)
    return predictions, max_softmax


def get_onthefly_dataloader(
    images: Sequence[tuple[int, int]] | np.ndarray,
    id_images_paths: list[Path],
    labels: Sequence | np.ndarray | None = None,
) -> DataLoaderWithLabels:
    """This dataloader will load images from disk "on the fly" when asked in
    every batch. It is fast due to PyTorch parallelization with `num_workers`
    and it is very RAM efficient. Only recommended to use in predictions.
    For training it is best to use preloaded images."""
    logging.info("Creating test IdentificationDataset with %d images", len(images))
    return DataLoader(
        SimpleDataset(images, labels),
        conf.BATCH_SIZE_PREDICTIONS,
        num_workers=2 if os.name == "nt" else 4,  # windows
        persistent_workers=True,
        collate_fn=partial(collate_fun, id_images_paths=id_images_paths),
        pin_memory=True,
    )


def collate_fun(
    locations_and_labels: list[tuple[tuple[int, int], int]], id_images_paths: list[Path]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Receives the batch images locations (episode and index).
    These are used to load the images and generate the batch tensor"""
    locations, labels = zip(*locations_and_labels)
    return (
        torch.from_numpy(load_id_images(id_images_paths, locations, verbose=False))
        .type(torch.float32)
        .unsqueeze(1),
        torch.tensor(labels),
    )


class SimpleDataset(Dataset):
    def __init__(
        self, images: Sequence | np.ndarray, labels: Sequence | np.ndarray | None = None
    ):
        super().__init__()
        self.images = images
        if labels is not None:
            self.labels = np.asarray(labels).astype(np.int64)
        else:
            self.labels = np.full(len(images), -1, np.int64)
        assert self.labels.ndim == 1
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]
