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

from idtrackerai.network.evaluate import evaluate
from idtrackerai.network.learners.learners import LearnerClassification
from idtrackerai.network.train import train

from ..network.network_params_crossings import NetworkParamsCrossings
from ..network.stop_training_criteria_crossings import StopTraining


def train_deep_crossing(
    learner: LearnerClassification,
    train_loader,
    val_loader,
    network_params: NetworkParamsCrossings,
    stop_training: StopTraining,
) -> tuple[bool, Path]:
    logging.info("Training Deep Crossing Detector")

    # Initialize metric storage
    train_losses = []
    if network_params.loss in ("CEMCL", "CEMCL_weighted"):
        train_losses_CE = []
        train_losses_MCL = []
        val_losses_CE = []
        val_losses_MCL = []
    train_accs = []
    val_losses = []
    val_accs = []

    logging.debug("Entering the epochs loop...")
    with Console().status("[red]Epochs loop...") as status:
        while not stop_training(train_losses, val_losses, val_accs, status):
            epoch = stop_training.epochs_completed
            (loss, loss_CE, loss_MCL), train_acc = train(
                epoch, train_loader, learner, network_params
            )

            train_losses.append(loss)
            if network_params.loss in ("CEMCL", "CEMCL_weighted"):
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
                if network_params.loss in ("CEMCL", "CEMCL_weighted"):
                    val_losses_CE.append(loss_CE)
                    val_losses_MCL.append(loss_MCL)
                val_accs.append(val_acc)
            # Save checkpoint at each LR steps and the end of optimization
            best_model_path = learner.snapshot(
                network_params.save_folder
                / f"{network_params.dataset}_{network_params.model_name}_{network_params.saveid}"
            )
            try:
                status.update(
                    f"[red]Epochs loop {epoch}: training loss ="
                    f" {train_losses[-1]:.6f}, validation loss ="
                    f" {val_losses[-1]:.6f} and accuracy = {val_accs[-1]:.4%}"
                )
            except IndexError:
                pass

        logging.info("Last epoch loop: %s", status.status, extra={"markup": True})

    return np.isnan(train_losses[-1]) or np.isnan(val_losses[-1]), best_model_path
