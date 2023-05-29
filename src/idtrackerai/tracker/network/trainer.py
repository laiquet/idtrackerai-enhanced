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
from contextlib import suppress

import numpy as np
from rich.console import Console
from torch.utils.data import DataLoader

from idtrackerai.network import LearnerClassification, NetworkParams, evaluate, train
from idtrackerai.utils import CustomError

from .stop_training_criteria import StopTraining


def TrainIdentification(
    learner: LearnerClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    network_params: NetworkParams,
    stop_training: StopTraining,
):
    logging.info("Training Identification Network")

    # Initialize metric storage
    train_loss = 0.0
    val_losses = []
    val_acc = 0.0

    logging.debug("Entering the epochs loop...")
    with Console().status("[red]Epochs loop...") as status:
        while not stop_training(train_loss, val_losses, val_acc, status):
            epoch = stop_training.epochs_completed

            train_loss, train_acc = train(epoch, train_loader, learner, network_params)
            val_loss, val_acc = evaluate(val_loader, network_params, learner)

            val_losses.append(val_loss)

            with suppress(IndexError):
                status.update(
                    f"[red]Epoch {epoch}: training loss = {train_loss:.6f},"
                    f" validation loss = {val_loss:.6f} and accuracy ="
                    f" {val_acc:.4%}"
                )

        logging.info("Last epoch loop: %s", status.status, extra={"markup": True})

    learner.save_model(network_params.model_path, val_acc=val_acc)

    if np.isnan(train_loss) or np.isnan(val_loss):
        raise CustomError("The model diverged")

    logging.info("Identification network trained")
