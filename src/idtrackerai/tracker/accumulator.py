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
from shutil import copyfile

import torch
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import Video
from idtrackerai.network import LearnerClassification, NetworkParams, get_device
from idtrackerai.utils import conf

from .accumulation_manager import (
    AccumulationManager,
    get_predictions_of_candidates_fragments,
)
from .identity_dataset import get_training_data_loaders, split_data_train_and_validation
from .identity_network import StopTraining, TrainIdentification


def perform_one_accumulation_step(
    accumulation_manager: AccumulationManager,
    video: Video,
    identification_model: Module,
    network_params: NetworkParams,
):
    logging.info(
        f"[bold]Performing new accumulation, step {accumulation_manager.current_step}",
        extra={"markup": True},
    )
    video.accumulation_step = accumulation_manager.current_step

    # Get images for training
    accumulation_manager.get_new_images_and_labels()
    images, labels = accumulation_manager.get_images_and_labels_for_training()
    train_data, val_data = split_data_train_and_validation(
        images, labels, validation_proportion=conf.VALIDATION_PROPORTION
    )
    assert images.shape[0] == labels.shape[0]
    logging.info(
        "Training with %d, validating with %d",
        len(train_data["images"]),
        len(val_data["images"]),
    )
    assert len(val_data["images"]) > 0

    # Set data loaders
    train_loader, val_loader = get_training_data_loaders(
        video.number_of_animals, train_data, val_data
    )

    # Set criterion
    logging.info("Setting training criterion")
    criterion = CrossEntropyLoss(weight=torch.tensor(train_data["weights"]))

    logging.info("Sending model and criterion to GPU")
    cudnn.benchmark = True  # make it train faster
    identification_model = identification_model.to(get_device())
    criterion = criterion.to(get_device())

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

    # Set scheduler
    logging.info("Setting scheduler")
    scheduler = MultiStepLR(optimizer, milestones=network_params.schedule, gamma=0.1)

    learner = LearnerClassification(
        identification_model, criterion, optimizer, scheduler
    )

    stop_training = StopTraining(
        network_params.number_of_classes,
        is_first_accumulation=video.accumulation_step == 0,
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
    accumulation_manager.update_set_of_individual_fragments_used()

    # compute ratio of accumulated images and stop if it is above random
    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
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
        for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
    ):
        logging.info(
            "Generating [bold]predictions[/bold] on remaining global fragments",
            extra={"markup": True},
        )
        (
            predictions,
            softmax_probs,
            indices_to_split,
            candidate_individual_fragments_identifiers,
        ) = get_predictions_of_candidates_fragments(
            identification_model,
            video.id_images_file_paths,
            network_params,
            accumulation_manager.list_of_fragments,
        )

        accumulation_manager.split_predictions_after_network_assignment(
            predictions,
            softmax_probs,
            indices_to_split,
            candidate_individual_fragments_identifiers,
        )
        # assign identities to the global fragments based on the predictions
        logging.info(
            "Checking eligibility criteria and generate the "
            "new list of identified global fragments to accumulate"
        )
        accumulation_manager.get_acceptable_global_fragments_for_training(
            candidate_individual_fragments_identifiers, video.accumulation_trial
        )

        accumulation_manager.print_accumulation_variables()

        stats = video.accumulation_statistics

        stats["n_accumulated_global_fragments"].append(
            sum(
                global_fragment.used_for_training
                for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
            )
        )
        stats["n_non_certain_global_fragments"].append(
            accumulation_manager.number_of_noncertain_global_fragments
        )
        stats["n_randomly_assigned_global_fragments"].append(
            accumulation_manager.number_of_random_assigned_global_fragments
        )
        stats["n_nonconsistent_global_fragments"].append(
            accumulation_manager.number_of_nonconsistent_global_fragments
        )
        stats["n_nonunique_global_fragments"].append(
            accumulation_manager.number_of_nonunique_global_fragments
        )
        stats["n_acceptable_global_fragments"].append(
            sum(
                global_fragment.acceptable_for_training(
                    accumulation_manager.accumulation_strategy
                )
                for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
            )
        )
        stats["ratio_of_accumulated_images"].append(
            accumulation_manager.ratio_accumulated_images
        )

        accumulation_manager.current_step += 1

    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
    )

    video.accumulation_statistics_data[video.accumulation_trial] = (
        video.accumulation_statistics
    )
