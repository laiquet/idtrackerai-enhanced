import json
import logging
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.sgd import SGD

from idtrackerai import Session, conf
from idtrackerai.utils import load_id_images

from ..network import (
    DEVICE,
    DataLoaderWithLabels,
    IdCNN,
    IdentifierIdCNN,
    StopTraining,
    evaluate_only_acc,
    get_dataloader,
    get_onthefly_dataloader,
    train_loop,
)
from .accumulation_manager import (
    AccumulationManager,
    get_predictions_of_candidates_fragments,
)


def accumulation_step(
    accumulation_manager: AccumulationManager,
    session: Session,
    identification_model: IdCNN,
    model_path: Path,
    penultimate_model_path: Path,
) -> bool:
    logging.info(
        f"[bold]Performing new accumulation, step {accumulation_manager.current_step}",
        extra={"markup": True},
    )
    id_img_paths = session.id_images_file_paths

    # Get images for training
    accumulation_manager.get_new_images_and_labels()

    # get a mixture of old and new images to train
    images_for_training, labels_for_training = (
        accumulation_manager.get_old_and_new_images()
    )

    train_dataloader, val_dataloader, weights = split_data_and_get_dataloaders(
        load_id_images(id_img_paths, images_for_training),
        labels_for_training,
        session.n_animals,
    )

    optimizer = SGD(identification_model.parameters(), lr=0.005, momentum=0.9)

    stopping = StopTraining(
        epochs_limit=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
        overfitting_limit=(
            conf.OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM
            if accumulation_manager.current_step == 0
            else conf.OVERFITTING_COUNTER_THRESHOLD_IDCNN
        ),
        plateau_limit=conf.LEARNING_RATIO_DIFFERENCE_IDCNN,
    )

    criterion = CrossEntropyLoss(weights).to(DEVICE)
    if conf.TORCH_COMPILE:
        criterion.compile()

    train_loop(
        model=identification_model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        stop_training=stopping,
        scheduler=MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1),
    )

    accumulation_manager.update_fragments_used_for_training()
    accumulation_manager.update_used_images_and_labels()
    accumulation_manager.assign_identities_to_fragments_used_for_training()

    # compute ratio of accumulated images and stop if it is above random
    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.ratio_of_images_used_for_training
    )

    test_acc = test_model(accumulation_manager, id_img_paths, identification_model)

    # keep a copy of the penultimate model
    penultimate_model_path.unlink(missing_ok=True)
    if model_path.is_file():
        copyfile(model_path, penultimate_model_path)
        copyfile(
            model_path.with_suffix(".metadata.json"),
            penultimate_model_path.with_suffix(".metadata.json"),
        )

    logging.info("Saving model at %s", model_path)
    torch.save(identification_model.state_dict(), model_path)
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "test_acc": test_acc,
                "ratio_accumulated": accumulation_manager.ratio_accumulated_images,
            },
            indent=4,
        )
    )

    if (
        accumulation_manager.ratio_accumulated_images
        > conf.THRESHOLD_EARLY_STOP_ACCUMULATION
    ):
        logging.info(
            "The ratio of accumulated images is higher than"
            f" {conf.THRESHOLD_EARLY_STOP_ACCUMULATION:.1%}, [bold]stopping"
            " accumulation by early stopping criteria",
            extra={"markup": True},
        )
        return True

    # Set accumulation parameters for rest of the accumulation
    # take images from global fragments not used in training (in the remainder test global fragments)
    if any(
        not global_fragment.used_for_training
        for global_fragment in accumulation_manager.list_of_global_fragments
    ):
        logging.info(
            "Generating [bold]predictions[/bold] on remaining global fragments",
            extra={"markup": True},
        )
        (
            predictions,
            probabilities,
            indices_to_split,
            candidate_fragments_identifiers,
        ) = get_predictions_of_candidates_fragments(
            IdentifierIdCNN(identification_model),
            id_img_paths,
            accumulation_manager.list_of_fragments,
        )

        accumulation_manager.split_predictions_after_network_assignment(
            predictions,
            probabilities,
            indices_to_split,
            candidate_fragments_identifiers,
        )

        accumulation_manager.assign_identities()
        accumulation_manager.update_accumulation_statistics()
        accumulation_manager.current_step += 1

    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.ratio_of_images_used_for_training
    )

    session.accumulation_statistics_data = accumulation_manager.accumulation_statistics
    return False


def test_model(
    accumulation_manager: AccumulationManager, id_img_paths: list[Path], model: IdCNN
):
    """Takes a sample of the accumulated images to test model's accuracy"""
    logging.info(
        "Using a sample of all accumulated images to test model's overall accuracy"
    )
    test_images, test_labels = accumulation_manager.get_old_images()
    test_dataloader = get_onthefly_dataloader(test_images, id_img_paths, test_labels)
    test_acc = evaluate_only_acc(test_dataloader, model)
    logging.info(f"Current model has an overall accuracy of {test_acc:.3%}")
    return test_acc


def split_data_and_get_dataloaders(
    images: np.ndarray,
    labels: np.ndarray,
    n_animals: int,
    validation_proportion: float = conf.VALIDATION_PROPORTION,
) -> tuple[DataLoaderWithLabels, DataLoaderWithLabels, torch.Tensor]:
    """Splits a set of `images` and `labels` into training and validation Dataloaders

    Parameters
    ----------
    images : np.ndarray
        List of images (arrays of shape [height, width])
    labels : np.ndarray
        List of integers from 0 to `number_of_animals` - 1
    validation_proportion : float
        The proportion of images that will be used to create the validation set.

    Returns
    -------
    train_dataloader : DataLoaderWithLabels
    val_dataloader : DataLoaderWithLabels
    weights : torch.Tensor
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

    for label in np.unique(labels):
        indices = labels == label
        images_per_class = images[indices]
        labels_per_class = labels[indices]
        n_images_validation = int(validation_proportion * len(labels_per_class) + 0.99)

        validation_images.append(images_per_class[:n_images_validation])
        validation_labels.append(labels_per_class[:n_images_validation])
        train_images.append(images_per_class[n_images_validation:])
        train_labels.append(labels_per_class[n_images_validation:])

    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    validation_images = np.concatenate(validation_images)
    validation_labels = np.concatenate(validation_labels)

    assert len(train_images) == len(train_labels)
    assert len(validation_images) > 0

    train_dataloader = get_dataloader(
        "training", train_images, train_labels, conf.BATCH_SIZE_IDCNN
    )
    val_dataloader = get_dataloader("validation", validation_images, validation_labels)

    weights = 1 - torch.bincount(
        torch.from_numpy(train_labels), minlength=n_animals
    ).type(torch.float32) / len(train_labels)

    return train_dataloader, val_dataloader, weights
