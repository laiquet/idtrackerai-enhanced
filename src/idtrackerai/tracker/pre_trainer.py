import logging
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import GlobalFragment
from idtrackerai.network import (
    CNN,
    DEVICE,
    LearnerClassification,
    NetworkParams,
    fully_connected_reinitialization,
)
from idtrackerai.utils import conf, load_id_images

from .identity_dataset import get_identity_dataloader, split_data_train_and_validation
from .identity_network import StopTraining, train_identification


def pretrain_global_fragment(
    identification_model: CNN,
    network_params: NetworkParams,
    pretraining_global_fragment: GlobalFragment,
    id_images_file_paths: list[Path],
):
    """Performs pretraining on a single global fragments"""

    images, labels = pretraining_global_fragment.get_images_and_labels()

    images = load_id_images(id_images_file_paths, images)

    (
        train_images,
        train_labels,
        train_weights,
        validation_images,
        validation_labels,
    ) = split_data_train_and_validation(
        images, labels, conf.VALIDATION_PROPORTION, network_params.n_classes
    )

    train_loader = get_identity_dataloader("training", train_images, train_labels)
    val_loader = get_identity_dataloader(
        "validation", validation_images, validation_labels
    )

    criterion = CrossEntropyLoss(weight=torch.from_numpy(train_weights))

    identification_model.apply(fully_connected_reinitialization)

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

    stop_training = StopTraining(network_params.n_classes)

    train_identification(learner, train_loader, val_loader, stop_training)
    learner.save_model(network_params.model_path)

    for fragment in pretraining_global_fragment:
        fragment.used_for_pretraining = True
