import logging
import os
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from itertools import count
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
from rich.console import Console
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import VisionDataset

from idtrackerai.utils import conf, load_id_images, track

from . import DEVICE, IdCNN, IdentifierBase


class DataLoaderWithLabels(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]: ...


NUMBER_OF_PIN_MEMORY_USED = 0


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
    model: IdCNN,
    criterion: CrossEntropyLoss,
    optimizer: Optimizer,
    train_loader: DataLoaderWithLabels,
    val_loader: DataLoaderWithLabels,
    stop_training: Callable[[float, float, float], str],
    scheduler: LRScheduler | None = None,
):
    logging.debug("Entering the training loop...")
    with Console().status("[red]Initializing training...") as status:
        for epoch in count(1):
            train_loss = train(train_loader, model, criterion, optimizer, scheduler)
            val_loss, val_acc = evaluate(val_loader, model, criterion)

            status.update(
                f"[red]Epoch {epoch:2}: training_loss={train_loss:.5f}, "
                f"validation_loss={val_loss:.5f} and accuracy={val_acc:.3%}"
            )
            stop_message = stop_training(train_loss, val_loss, val_acc)
            if stop_message:
                break
        else:
            raise

    logging.info(stop_message)
    logging.info("Last epoch: %s", status.status, extra={"markup": True})
    logging.info("Network trained")


def train(
    train_loader: DataLoaderWithLabels,
    model: IdCNN,
    criterion: CrossEntropyLoss,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> float:
    """Trains trains a network using a learner, a given train_loader"""
    losses = 0
    n_predictions = 0

    model.train()

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        out = model(images)
        loss = criterion(out, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses += loss.item() * len(images)
        n_predictions += len(images)

    if scheduler is not None:
        scheduler.step()
    return losses / n_predictions


@torch.inference_mode()
def evaluate(
    eval_loader: DataLoaderWithLabels, model: IdCNN, criterion: CrossEntropyLoss
) -> tuple[float, float]:
    losses = 0
    n_predictions = 0
    n_right_guess = 0

    model.eval()

    for images, labels in eval_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        out = model(images)
        loss = criterion(out, labels)
        n_predictions += len(labels)
        n_right_guess += (out.max(1).indices == labels).count_nonzero().item()

        losses += loss.item() * len(images)

    return losses / n_predictions, n_right_guess / n_predictions


@torch.inference_mode()
def evaluate_only_acc(eval_loader: DataLoaderWithLabels, model: IdCNN) -> float:
    model.eval()
    n_predictions = 0
    n_right_guess = 0

    for images, labels in eval_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        predictions = model(images).max(1).indices
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
    global NUMBER_OF_PIN_MEMORY_USED
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

    # We set pin_memory on training only because of https://github.com/pytorch/pytorch/issues/91252
    # And we limit the number of dataloaders created with pin_memory
    pin_memory = scope == "training"
    if NUMBER_OF_PIN_MEMORY_USED > 5:
        pin_memory = False
    if pin_memory:
        NUMBER_OF_PIN_MEMORY_USED += 1
    # logging.debug(f"{pin_memory=}")

    return DataLoader(
        ImageDataset(images, labels, transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=scope == "training",
        num_workers=1 if os.name == "nt" else 4,  # windows
        persistent_workers=True,
        pin_memory=pin_memory,
    )


@torch.inference_mode()
def get_predictions(
    model: IdentifierBase,
    image_location: Sequence[tuple[int, int]] | np.ndarray,
    id_images_paths: Sequence[Path],
    kind: str = "identities",
) -> tuple[np.ndarray, np.ndarray]:
    logging.debug("Predicting %s of %d images", kind, len(image_location), stacklevel=2)
    predictions = np.empty(len(image_location), np.int32)
    probabilities = np.empty(len(image_location), np.float32)
    index = 0
    model.to(DEVICE)
    model.eval()
    dataloader = get_onthefly_dataloader(image_location, id_images_paths)
    for images, _labels in track(dataloader, "Predicting " + kind):
        classification_probs = model(images.to(DEVICE))
        batch_probabilities, batch_predictions = classification_probs.max(dim=1)
        batch_predictions += 1
        batch_size = len(batch_predictions)
        predictions[index : index + batch_size] = batch_predictions.numpy(force=True)
        probabilities[index : index + batch_size] = batch_probabilities.numpy(
            force=True
        )
        index += batch_size
    assert index == len(predictions) == len(probabilities)
    return predictions, probabilities


def get_onthefly_dataloader(
    image_locations: Sequence[tuple[int, int]] | np.ndarray,
    id_images_paths: Sequence[Path],
    labels: Sequence | np.ndarray | None = None,
) -> DataLoaderWithLabels:
    """This dataloader will load images from disk "on the fly" when asked in
    every batch. It is fast due to PyTorch parallelization with `num_workers`
    and it is very RAM efficient. Only recommended to use in predictions.
    For training it is best to use preloaded images."""
    logging.info("Creating on-the-fly DataLoader with %d images", len(image_locations))
    return DataLoader(
        SimpleDataset(image_locations, labels),
        conf.BATCH_SIZE_PREDICTIONS,
        num_workers=2,
        collate_fn=partial(collate_fun, id_images_paths=id_images_paths),
        # pin_memory=True, https://github.com/pytorch/pytorch/issues/91252
    )


def collate_fun(
    locations_and_labels: list[tuple[tuple[int, int], int]],
    id_images_paths: Sequence[Path],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Receives the batch images locations (episode and index).
    These are used to load the images and generate the batch tensor"""
    locations, labels = zip(*locations_and_labels)
    return (
        torch.from_numpy(
            load_id_images(id_images_paths, locations, verbose=False, dtype=np.float32)
        ).unsqueeze(1)
        / 255,
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
