from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import GlobalFragment
from idtrackerai.utils import conf, load_id_images

from ..network import (
    CNN,
    DEVICE,
    LearnerClassification,
    NetworkParams,
    StopTraining,
    get_dataloader,
    train_loop,
)
from .identity_dataset import split_data_train_and_validation


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

    train_loader = get_dataloader(
        "training", train_images, train_labels, conf.BATCH_SIZE_IDCNN
    )
    val_loader = get_dataloader("validation", validation_images, validation_labels)

    criterion = CrossEntropyLoss(
        weight=torch.tensor(train_weights, dtype=torch.float32)
    ).to(DEVICE)

    identification_model.fully_connected_reinitialization()

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

    stopping = StopTraining(
        epochs_limit=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
        overfitting_limit=conf.OVERFITTING_COUNTER_THRESHOLD_IDCNN,
        plateau_limit=conf.LEARNING_RATIO_DIFFERENCE_IDCNN,
    )

    train_loop(learner, train_loader, val_loader, stopping)
    learner.save_model(network_params.model_path)

    for fragment in pretraining_global_fragment:
        fragment.used_for_pretraining = True
