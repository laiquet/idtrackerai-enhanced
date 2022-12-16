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

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import Video
from idtrackerai.utils import conf

from .accumulation_manager import (
    AccumulationManager,
    get_predictions_of_candidates_fragments,
)
from .dataset.identification_dataloader import get_training_data_loaders
from .dataset.identification_dataset import split_data_train_and_validation
from .network.network_params import NetworkParams
from .network.stop_training_criteria import Stop_Training
from .network.trainer import TrainIdentification


def perform_one_accumulation_step(
    accumulation_manager: AccumulationManager,
    video: Video,
    identification_model: nn.Module,
    learner_class: nn.Module,
    network_params: NetworkParams,
):

    # Set accumulation counter
    logging.info(f"Accumulation step {accumulation_manager.counter}")
    video.accumulation_step = accumulation_manager.counter

    # Get images for training
    accumulation_manager.get_new_images_and_labels()
    images, labels = accumulation_manager.get_images_and_labels_for_training()
    train_data, val_data = split_data_train_and_validation(
        images, labels, validation_proportion=conf.VALIDATION_PROPORTION
    )
    assert images.shape[0] == labels.shape[0]
    logging.debug(
        f"{images.shape[0]} labeled images with shape {images.shape[1:]}, "
        f"training with {len(train_data['images'])} and "
        f"validating with {len(val_data['images'])}"
    )
    assert len(val_data["images"]) > 0

    # Set data loaders
    train_loader, val_loader = get_training_data_loaders(
        video.number_of_animals, train_data, val_data
    )

    # Set criterion
    logging.info("Setting training criterion")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_data["weights"]))

    # Send model and criterion to GPU
    if network_params.use_gpu:
        logging.info("Sending model and criterion to GPU")
        torch.cuda.set_device(0)
        cudnn.benchmark = True  # make it train faster
        identification_model = identification_model.cuda()
        criterion = criterion.cuda()

    # Set optimizer
    logging.info("Setting optimizer")
    optimizer = torch.optim.__dict__[network_params.optimizer](
        identification_model.parameters(), **network_params.optim_args
    )

    # Set scheduler
    logging.info("Setting scheduler")
    scheduler = MultiStepLR(
        optimizer, milestones=network_params.schedule, gamma=0.1
    )

    # Set learner
    logging.info("Setting the learner")
    learner = learner_class(
        identification_model, criterion, optimizer, scheduler
    )

    # Set stopping criteria
    logging.info("Setting the stopping criteria")
    # set criteria to stop the training
    stop_training = Stop_Training(
        network_params.number_of_classes,
        check_for_loss_plateau=True,
        first_accumulation_flag=video is None or video.accumulation_step == 0,
    )

    trainer = TrainIdentification(
        learner,
        train_loader,
        val_loader,
        network_params,
        stop_training,
        accumulation_manager=accumulation_manager,
    )
    logging.info("Identification network trained")

    # update the set of images used for training
    logging.info("Update images and labels used for training")
    accumulation_manager.update_used_images_and_labels()

    # assign identities fo the global fragments that have been used for training
    logging.info("Assigning identities to accumulated global fragments")
    accumulation_manager.assign_identities_to_fragments_used_for_training()

    # update the list of individual fragments that have been used for training
    logging.info("Update list of individual fragments used for training")
    accumulation_manager.update_list_of_individual_fragments_used()

    # compute ratio of accumulated images and stop if it is above random
    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
    )
    logging.info(
        f"The {accumulation_manager.ratio_accumulated_images:.3%} of the "
        "images have been accumulated"
    )
    if (
        accumulation_manager.ratio_accumulated_images
        > conf.THRESHOLD_EARLY_STOP_ACCUMULATION
    ):
        logging.debug("Stopping accumulation by early stopping criteria")
        return accumulation_manager.ratio_accumulated_images

    # Set accumulation parameters for rest of the accumulation
    # take images from global fragments not used in training (in the remainder test global fragments)
    logging.info("Get new global fragments for training")
    if any(
        [
            not global_fragment.used_for_training
            for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
        ]
    ):
        logging.info("Generate predictions on candidate global fragments")
        (
            predictions,
            softmax_probs,
            indices_to_split,
            candidate_individual_fragments_identifiers,
        ) = get_predictions_of_candidates_fragments(
            identification_model,
            video,
            network_params,
            accumulation_manager.list_of_fragments.fragments,
        )
        logging.debug("Splitting predictions by fragments...")
        accumulation_manager.split_predictions_after_network_assignment(
            predictions,
            softmax_probs,
            indices_to_split,
            candidate_individual_fragments_identifiers,
        )
        # assign identities to the global fragments based on the predictions
        logging.info(
            "Checking eligibility criteria and generate the new list of global fragments to accumulate"
        )
        accumulation_manager.get_acceptable_global_fragments_for_training(
            candidate_individual_fragments_identifiers
        )
        # Million logs

        logging.info(
            "Number of non certain global fragments: %i"
            % accumulation_manager.number_of_noncertain_global_fragments
        )
        logging.info(
            "Number of randomly assigned global fragments: %i"
            % accumulation_manager.number_of_random_assigned_global_fragments
        )
        logging.info(
            "Number of non consistent global fragments: %i "
            % accumulation_manager.number_of_nonconsistent_global_fragments
        )
        logging.info(
            "Number of non unique global fragments: %i "
            % accumulation_manager.number_of_nonunique_global_fragments
        )
        logging.info(
            "Number of acceptable global fragments: %i "
            % accumulation_manager.number_of_acceptable_global_fragments
        )
        logging.info(
            "Number of non certain fragments: %i"
            % accumulation_manager.number_of_noncertain_fragments
        )
        logging.info(
            "Number of randomly assigned fragments: %i"
            % accumulation_manager.number_of_random_assigned_fragments
        )
        logging.info(
            "Number of non consistent fragments: %i "
            % accumulation_manager.number_of_nonconsistent_fragments
        )
        logging.info(
            "Number of non unique fragments: %i "
            % accumulation_manager.number_of_nonunique_fragments
        )
        logging.info(
            "Number of acceptable fragments: %i "
            % accumulation_manager.number_of_acceptable_fragments
        )

        new_values = [
            len(
                [
                    global_fragment
                    for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
                    if global_fragment.used_for_training
                ]
            ),
            accumulation_manager.number_of_noncertain_global_fragments,
            accumulation_manager.number_of_random_assigned_global_fragments,
            accumulation_manager.number_of_nonconsistent_global_fragments,
            accumulation_manager.number_of_nonunique_global_fragments,
            np.count_nonzero(
                [
                    global_fragment.acceptable_for_training(
                        accumulation_manager.accumulation_strategy
                    )
                    for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
                ]
            ),
            accumulation_manager.ratio_accumulated_images,
        ]
        video.store_accumulation_step_statistics_data(new_values)
        accumulation_manager.update_counter()

    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
    )
    video.store_accumulation_statistics_data(video.accumulation_trial)
    return accumulation_manager.ratio_accumulated_images
