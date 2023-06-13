import logging

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import VisionDataset

from idtrackerai.network import normalize
from idtrackerai.utils import conf

num_workers_train = 1
num_workers_val = 1


class IdentificationDataset(VisionDataset):
    def __init__(self, data_dict, scope, transform=None):
        super().__init__("", transform=transform)
        self.scope = scope
        self.images = data_dict["images"]
        if self.scope in ("training", "validation", "test"):
            self.labels = data_dict["labels"]
        else:
            self.labels = np.zeros((self.images.shape[0]))
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


def split_data_train_and_validation(images, labels, validation_proportion=None):
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

    if validation_proportion is None:
        validation_proportion = conf.VALIDATION_PROPORTION

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
    number_of_animals: int, train_data, val_data
) -> tuple[DataLoader, DataLoader]:
    logging.info("Creating training IdentificationDataset")
    training_set = IdentificationDataset(
        train_data,
        scope="training",
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    train_loader = DataLoader(
        training_set,
        batch_size=conf.BATCH_SIZE_IDCNN,
        shuffle=True,
        num_workers=num_workers_train,
        persistent_workers=num_workers_train > 0,
    )
    train_loader.num_classes = number_of_animals
    train_loader.image_shape = training_set[0][0].shape

    logging.info("Creating validation IdentificationDataset")
    validation_set = IdentificationDataset(
        val_data,
        scope="validation",
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=conf.BATCH_SIZE_PREDICTIONS_IDCNN,
        num_workers=num_workers_val,
        persistent_workers=num_workers_val > 0,
    )
    val_loader.num_classes = number_of_animals
    val_loader.image_shape = validation_set[0][0].shape
    return train_loader, val_loader


def get_test_data_loader(test_data, number_of_classes):
    logging.debug("Creating test IdentificationDataset")
    test_set = IdentificationDataset(
        test_data,
        scope="predict",
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=conf.BATCH_SIZE_PREDICTIONS_IDCNN,
        num_workers=num_workers_val,
        persistent_workers=num_workers_val > 0,
    )
    test_loader.num_classes = number_of_classes
    test_loader.image_shape = test_set[0][0].shape
    return test_loader
