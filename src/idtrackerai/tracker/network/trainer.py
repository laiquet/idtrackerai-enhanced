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

import numpy as np
from rich.console import Console
from torch.utils.data import DataLoader

from idtrackerai.network.evaluate import evaluate
from idtrackerai.network.learners import Learner_Classification
from idtrackerai.network.train import train

from ..accumulation_manager import AccumulationManager
from .network_params import NetworkParams
from .stop_training_criteria import Stop_Training


def TrainIdentification(
    learner: Learner_Classification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    network_params: NetworkParams,
    stop_training: Stop_Training,
    accumulation_manager: AccumulationManager | None = None,
) -> Path:
    logging.info("Training Identification Network")
    # TODO: Store accuracies and losses
    # store_training_accuracy_and_loss_data = \
    #     Store_Accuracy_and_Loss(
    #         self.network_params.save_folder,name='training')
    # store_validation_accuracy_and_loss_data = \
    #     Store_Accuracy_and_Loss(
    #         self.network_params.save_folder, ame='validation')

    # Initialize metric storage
    train_losses = []
    if network_params.loss in ["CEMCL", "CEMCL_weighted"]:
        train_losses_CE = []
        train_losses_MCL = []
        val_losses_CE = []
        val_losses_MCL = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = -1
    logging.debug("entering the epochs loop...")
    with Console().status("[red]Epochs loop...") as status:
        while not stop_training(train_losses, val_losses, val_accs, status):
            epoch = stop_training.epochs_completed
            status.update(f"[red]Epochs loop (epoch {epoch})...")
            (loss, loss_CE, loss_MCL), train_acc = train(
                epoch, train_loader, learner, network_params
            )

            train_losses.append(loss)
            if network_params.loss in ["CEMCL", "CEMCL_weighted"]:
                train_losses_CE.append(loss_CE)
                train_losses_MCL.append(loss_MCL)
            train_accs.append(train_acc)

            if val_loader is not None and (
                (not network_params.skip_eval) or (epoch == network_params.epochs - 1)
            ):
                loss, loss_CE, loss_MCL, val_acc = evaluate(
                    val_loader, None, network_params, learner
                )
                val_losses.append(loss)
                if network_params.loss in ["CEMCL", "CEMCL_weighted"]:
                    val_losses_CE.append(loss_CE)
                    val_losses_MCL.append(loss_MCL)
                val_accs.append(val_acc)
            # Save checkpoint at each LR steps and the end of optimization
            ## TODO: Consider saving only best model
            best_model_path = learner.snapshot(network_params.save_model_path)

            best_val_acc = max(val_acc, best_val_acc)

    if np.isnan(train_losses[-1]) or np.isnan(val_losses[-1]):
        logging.warning(
            "The model diverged. Falling back to individual-crossing "
            "discrimination by average area model."
        )
    else:
        # update used_for_training flag to True for fragments used
        logging.info("Step completed.")
        if accumulation_manager is not None:
            accumulation_manager.update_fragments_used_for_training()

    return best_model_path
