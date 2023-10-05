import logging
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import VisionDataset

from idtrackerai.network import normalize
from idtrackerai.utils import conf

num_workers_train = 1
num_workers_val = 1


class IdentificationDataset(VisionDataset):
    def __init__(
        self,
        scope: Literal["training", "validation", "test", "predict"],
        images: np.ndarray,
        labels: np.ndarray | None = None,
        transform=None,
    ):
        super().__init__("", transform=transform)
        self.scope = scope
        self.images = images
        self.labels = labels if labels is not None else np.zeros((self.images.shape[0]))
        self.get_data()

    def get_data(self):
        if self.images.ndim <= 3:
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)

        if self.scope == "training":
            self.images, self.labels = duplicate_PCA_images(self.images, self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def split_data_train_and_validation(
    images: np.ndarray, labels: np.ndarray, validation_proportion: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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

    training_weights = (
        1.0 - np.unique(train_labels, return_counts=True)[1] / len(train_labels)
    ).astype("float32")

    train_dict = {
        "images": train_images,
        "labels": train_labels,
        "weights": training_weights,
    }
    val_dict = {"images": validation_images, "labels": validation_labels}
    return train_dict, val_dict


def duplicate_PCA_images(training_images, training_labels):
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


def get_training_data_loaders(
    train_data: dict[str, np.ndarray], val_data: dict[str, np.ndarray]
) -> tuple[DataLoader, DataLoader]:
    logging.info("Creating training IdentificationDataset")
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    training_set = IdentificationDataset(
        "training", train_data["images"], train_data["labels"], transform=transform
    )
    train_loader = DataLoader(
        training_set,
        batch_size=conf.BATCH_SIZE_IDCNN,
        shuffle=True,
        num_workers=num_workers_train,
        persistent_workers=num_workers_train > 0,
    )

    logging.info("Creating validation IdentificationDataset")
    validation_set = IdentificationDataset(
        "validation", val_data["images"], val_data["labels"], transform=transform
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=conf.BATCH_SIZE_PREDICTIONS_IDCNN,
        num_workers=num_workers_val,
        persistent_workers=num_workers_val > 0,
    )
    return train_loader, val_loader


def get_test_data_loader(images: np.ndarray):
    logging.debug("Generating prediction data set with %d images", len(images))
    test_set = IdentificationDataset(
        "predict",
        images,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    return DataLoader(
        test_set,
        batch_size=conf.BATCH_SIZE_PREDICTIONS_IDCNN,
        num_workers=num_workers_val,
        persistent_workers=num_workers_val > 0,
    )
