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
from statistics import fmean

import torch
from torch.utils.data import DataLoader

from . import LearnerClassification, NetworkParams
from .utils import Confusion


def train(
    epoch: int,
    train_loader: DataLoader,
    learner: LearnerClassification,
    network_params: NetworkParams,
):
    """Trains trains a network using a learner, a given train_loader and a set of network_params

    :param epoch: current epoch
    :param train_loader: dataloader
    :param learner: learner from learner.py
    :param network_params: networks params from networks_params.py
    :return: losses (tuple) and accuracy
    """

    # Initialize all meters
    losses = []
    confusion = Confusion(network_params.number_of_classes)

    # Setup learner's configuration
    learner.train()

    # The optimization loop
    for input_, target in train_loader:
        # Prepare the inputs
        if network_params.use_gpu:
            input_ = input_.cuda()
            target = target.cuda()
        train_target, eval_target = (target, target)

        # Optimization
        loss, output = learner.learn(input_, train_target)

        with torch.no_grad():
            confusion.add(output, eval_target)

        losses += [loss] * input_.size(0)

    learner.step_schedule(epoch)
    return fmean(losses), confusion.acc()
