# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)
import logging
from pathlib import Path

import torch
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import GlobalFragment
from idtrackerai.network import (
    DEVICE,
    LearnerClassification,
    NetworkParams,
    fc_weights_reinit,
)
from idtrackerai.utils import conf

from .identity_dataset import get_training_data_loaders, split_data_train_and_validation
from .identity_network import StopTraining, TrainIdentification


def pretrain_global_fragment(
    identification_model: Module,
    network_params: NetworkParams,
    pretraining_global_fragment: GlobalFragment,
    id_images_file_paths: list[Path],
):
    """Performs pretraining on a single global fragments"""

    images, labels = pretraining_global_fragment.get_images_and_labels(
        id_images_file_paths
    )

    train_data, val_data = split_data_train_and_validation(
        images, labels, validation_proportion=conf.VALIDATION_PROPORTION
    )
    logging.info(
        "Pretraining with %d images, %d for training and %d for validation",
        len(images),
        len(train_data["images"]),
        len(val_data["images"]),
    )

    # Set data loaders
    train_loader, val_loader = get_training_data_loaders(train_data, val_data)

    criterion = CrossEntropyLoss(weight=torch.tensor(train_data["weights"]))

    # Re-initialize fully-connected layers
    identification_model.apply(fc_weights_reinit)

    logging.info("Sending model and criterion to %s", DEVICE)
    cudnn.benchmark = True  # make it train faster
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

    stop_training = StopTraining(network_params.n_classes)

    TrainIdentification(
        learner, train_loader, val_loader, network_params, stop_training
    )

    for fragment in pretraining_global_fragment:
        fragment.used_for_pretraining = True
