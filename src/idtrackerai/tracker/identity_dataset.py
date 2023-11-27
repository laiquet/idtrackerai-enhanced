import logging
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import VisionDataset

from idtrackerai.network import DataLoaderWithLabels
from idtrackerai.utils import conf, load_id_images


class IdentificationDataset(VisionDataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        super().__init__("", transform=transform)
        self.images = images
        self.labels = labels.astype(np.int64)
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def split_data_train_and_validation(
    images: np.ndarray, labels: np.ndarray, validation_proportion: float, n_animals: int
) -> tuple[np.ndarray, ...]:
    """Splits a set of `images` and `labels` into training and validation sets

    Parameters
    ----------
    number_of_animals : int
        Number of classes in the set of images
    images : list
        List of images (arrays of shape [height, width])
    labels : list
        List of integers from 0 to `number_of_animals` - 1
    validation_proportion : float
        The proportion of images that will be used to create the validation set.


    Returns
    -------
    training_dataset : <DataSet object>
        Object containing the images and labels for training
    validation_dataset : <DataSet object>
        Object containing the images and labels for validation

    See Also
    --------
    :class:`get_data.DataSet`
    :func:`get_data.duplicate_PCA_images`
    """
    # Init variables
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []

    for i in np.unique(labels):
        # Get images of this individual
        this_indiv_images = images[labels == i]
        this_indiv_labels = labels[labels == i]
        # Compute number of images for training and validation
        num_images = len(this_indiv_labels)
        num_images_validation = np.ceil(validation_proportion * num_images).astype(int)
        num_images_training = num_images - num_images_validation
        # Get train, validation and test, images and labels
        train_images.append(this_indiv_images[:num_images_training])
        train_labels.append(this_indiv_labels[:num_images_training])
        validation_images.append(this_indiv_images[num_images_training:])
        validation_labels.append(this_indiv_labels[num_images_training:])

    train_images = np.vstack(train_images)
    train_labels = np.concatenate(train_labels, axis=0)

    validation_images = np.vstack(validation_images)
    validation_labels = np.concatenate(validation_labels, axis=0)

    train_weights = (
        1.0 - np.bincount(train_labels, minlength=n_animals) / len(train_labels)
    ).astype(np.float32)

    return (
        train_images,
        train_labels,
        train_weights,
        validation_images,
        validation_labels,
    )


def duplicate_PCA_images(training_images: np.ndarray, training_labels: np.ndarray):
    """Creates a copy of every image in `training_images` by rotating 180 degrees

    Parameters
    ----------
    training_images : ndarray
        Array of shape [number of images, height, width, channels] containing
        the images to be rotated
    training_labels : ndarray
        Array of shape [number of images, 1] containing the labels corresponding
        to the `training_images`

    Returns
    -------
    training_images : ndarray
        Array of shape [2*number of images, height, width, channels] containing
        the original images and the images rotated
    training_labels : ndarray
        Array of shape [2*number of images, 1] containing the labels corresponding
        to the original images and the images rotated
    """
    augmented_images = np.rot90(training_images, 2, axes=(1, 2))
    training_images = np.concatenate([training_images, augmented_images], axis=0)
    training_labels = np.concatenate([training_labels, training_labels], axis=0)
    return training_images, training_labels


def get_identity_dataloader(
    scope: Literal["training", "validation", "test"],
    images: np.ndarray,
    labels: np.ndarray | None = None,
) -> DataLoaderWithLabels:
    logging.info("Creating %s IdentificationDataset with %d images", scope, len(images))

    batch_size = (
        conf.BATCH_SIZE_IDCNN
        if scope == "training"
        else conf.BATCH_SIZE_PREDICTIONS_IDCNN
    )

    labels = labels if labels is not None else np.zeros(len(images))

    if scope == "training":
        images, labels = duplicate_PCA_images(images, labels)

    if images.ndim <= 3:
        images = np.expand_dims(images, axis=-1)

    dataset = IdentificationDataset(images, labels, transforms.ToTensor())
    return DataLoader(
        dataset,
        batch_size,
        shuffle=scope == "training",
        num_workers=1,
        persistent_workers=True,
    )


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
        conf.BATCH_SIZE_PREDICTIONS_IDCNN,
        num_workers=4,
        persistent_workers=True,
        collate_fn=partial(collate_fun, id_images_paths=id_images_paths),
    )


def collate_fun(
    locations_and_labels: list[tuple[tuple[int, int], int]], id_images_paths: list[Path]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Receives the batch images locations (episode and index).
    These are used to load the images and generate the batch tensor"""
    locations, labels = list(zip(*locations_and_labels))
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
