import numpy as np


def split_data_train_and_validation(
    images: np.ndarray, labels: np.ndarray, validation_proportion: float, n_animals: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    assert len(images) == len(labels)

    # shuffle images and labels
    shuffled_order = np.arange(len(images))
    np.random.shuffle(shuffled_order)
    images = images[shuffled_order]
    labels = labels[shuffled_order]

    # Init variables
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []

    for i in np.unique(labels):
        # Get images of this individual
        individual_indices = labels == i
        this_indiv_images = images[individual_indices]
        this_indiv_labels = labels[individual_indices]

        n_images_validation = int(validation_proportion * len(this_indiv_labels) + 0.99)

        validation_images.append(this_indiv_images[:n_images_validation])
        validation_labels.append(this_indiv_labels[:n_images_validation])
        train_images.append(this_indiv_images[n_images_validation:])
        train_labels.append(this_indiv_labels[n_images_validation:])

    train_labels = np.concatenate(train_labels)

    train_weights = 1.0 - np.bincount(train_labels, minlength=n_animals) / len(
        train_labels
    )

    return (
        np.concatenate(train_images),
        train_labels,
        train_weights,
        np.concatenate(validation_images),
        np.concatenate(validation_labels),
    )
