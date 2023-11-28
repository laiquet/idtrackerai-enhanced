import logging
from pathlib import Path
from shutil import copyfile

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import Session
from idtrackerai.network import (
    CNN,
    DEVICE,
    LearnerClassification,
    NetworkParams,
    evaluate_only_acc,
)
from idtrackerai.utils import conf, load_id_images

from .accumulation_manager import (
    AccumulationManager,
    get_predictions_of_candidates_fragments,
)
from .identity_dataset import get_identity_dataloader, split_data_train_and_validation
from .identity_network import StopTraining, train_identification


def perform_one_accumulation_step(
    accumulation_manager: AccumulationManager,
    session: Session,
    identification_model: CNN,
    network_params: NetworkParams,
):
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

    (
        train_images,
        train_labels,
        train_weights,
        validation_images,
        validation_labels,
    ) = split_data_train_and_validation(
        load_id_images(id_img_paths, images_for_training),
        labels_for_training,
        conf.VALIDATION_PROPORTION,
        session.n_animals,
    )
    assert len(images_for_training) == len(labels_for_training)
    assert len(validation_images) > 0

    train_loader = get_identity_dataloader("training", train_images, train_labels)
    val_loader = get_identity_dataloader(
        "validation", validation_images, validation_labels
    )

    criterion = CrossEntropyLoss(weight=torch.from_numpy(train_weights))

    logging.info("Sending model and criterion to %s", DEVICE)
    identification_model.to(DEVICE)
    criterion.to(DEVICE)

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

    train_identification(learner, train_loader, val_loader, stop_training)

    # free some RAM
    del train_loader, val_loader, train_images, validation_images

    accumulation_manager.update_fragments_used_for_training()
    accumulation_manager.update_used_images_and_labels()
    accumulation_manager.assign_identities_to_fragments_used_for_training()

    # compute ratio of accumulated images and stop if it is above random
    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.ratio_of_images_used_for_training
    )

    test_acc = test_model(accumulation_manager, id_img_paths, learner.model)

    # keep a copy of the penultimate model
    network_params.penultimate_model_path.unlink(missing_ok=True)
    if network_params.model_path.is_file():
        copyfile(network_params.model_path, network_params.penultimate_model_path)
    learner.save_model(
        network_params.model_path,
        test_acc=test_acc,
        ratio_accumulated=accumulation_manager.ratio_accumulated_images,
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
            identification_model, id_img_paths, accumulation_manager.list_of_fragments
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


def test_model(
    accumulation_manager: AccumulationManager, id_img_paths: list[Path], model: CNN
):
    """Takes a sample of the accumulated images to test model's accuracy"""
    logging.info(
        "Using a sample of all accumulated images to test model's overall accuracy"
    )
    test_images, test_labels = accumulation_manager.get_old_images()
    test_images = load_id_images(id_img_paths, test_images)
    test_dataloader = get_identity_dataloader("test", test_images, test_labels)
    test_acc = evaluate_only_acc(test_dataloader, model)
    logging.info(f"Current model has an overall accuracy of {test_acc:.3%}")
    return test_acc
