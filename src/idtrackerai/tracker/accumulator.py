import logging
from shutil import copyfile

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import Session
from idtrackerai.network import DEVICE, LearnerClassification, NetworkParams
from idtrackerai.utils import conf, load_id_images

from .accumulation_manager import (
    AccumulationManager,
    get_predictions_of_candidates_fragments,
)
from .identity_dataset import get_identity_dataloader, split_data_train_and_validation
from .identity_network import StopTraining, TrainIdentification


def perform_one_accumulation_step(
    accumulation_manager: AccumulationManager,
    session: Session,
    identification_model: Module,
    network_params: NetworkParams,
):
    logging.info(
        f"[bold]Performing new accumulation, step {accumulation_manager.current_step}",
        extra={"markup": True},
    )

    # Get images for training
    accumulation_manager.get_new_images_and_labels()
    images, labels = accumulation_manager.get_images_and_labels_for_training()
    images = load_id_images(session.id_images_file_paths, images)

    (
        train_images,
        train_labels,
        train_weights,
        validation_images,
        validation_labels,
    ) = split_data_train_and_validation(
        images, labels, validation_proportion=conf.VALIDATION_PROPORTION
    )
    assert len(images) == len(labels)
    assert len(validation_images) > 0

    train_loader = get_identity_dataloader("training", train_images, train_labels)
    val_loader = get_identity_dataloader(
        "validation", validation_images, validation_labels
    )

    criterion = CrossEntropyLoss(weight=torch.tensor(train_weights))

    logging.info("Sending model and criterion to %s", DEVICE)
    identification_model.to(DEVICE)
    criterion.to(DEVICE)

    logging.info(f"Setting {network_params.optimizer} optimizer")
    if network_params.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            identification_model.parameters(), **network_params.optim_args
        )
    elif network_params.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            identification_model.parameters(), **network_params.optim_args
        )
    else:
        raise AttributeError(network_params.optimizer)

    scheduler = MultiStepLR(optimizer, milestones=network_params.schedule, gamma=0.1)

    learner = LearnerClassification(
        identification_model, criterion, optimizer, scheduler
    )

    stop_training = StopTraining(
        network_params.n_classes,
        is_first_accumulation=accumulation_manager.current_step == 0,
    )

    # keep a copy of the penultimate model
    network_params.penultimate_model_path.unlink(missing_ok=True)
    if network_params.model_path.is_file():
        copyfile(network_params.model_path, network_params.penultimate_model_path)

    TrainIdentification(
        learner, train_loader, val_loader, network_params, stop_training
    )

    accumulation_manager.update_fragments_used_for_training()
    accumulation_manager.update_used_images_and_labels()
    accumulation_manager.assign_identities_to_fragments_used_for_training()

    # compute ratio of accumulated images and stop if it is above random
    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.ratio_of_images_used_for_training
    )

    if (
        accumulation_manager.ratio_accumulated_images
        > conf.THRESHOLD_EARLY_STOP_ACCUMULATION
    ):
        logging.debug("Stopping accumulation by early stopping criteria")
        return

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
            softmax_probs,
            indices_to_split,
            candidate_fragments_identifiers,
        ) = get_predictions_of_candidates_fragments(
            identification_model,
            session.id_images_file_paths,
            accumulation_manager.list_of_fragments,
        )

        accumulation_manager.split_predictions_after_network_assignment(
            predictions,
            softmax_probs,
            indices_to_split,
            candidate_fragments_identifiers,
        )

        accumulation_manager.assign_identities(session.accumulation_trial)
        accumulation_manager.update_accumulation_statistics()
        accumulation_manager.current_step += 1

    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.ratio_of_images_used_for_training
    )

    while len(session.accumulation_statistics_data) <= session.accumulation_trial:
        session.accumulation_statistics_data.append({})

    session.accumulation_statistics_data[session.accumulation_trial] = (
        accumulation_manager.accumulation_statistics
    )
