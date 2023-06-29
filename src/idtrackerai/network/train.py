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

from . import LearnerClassification, get_device


def train(epoch: int, train_loader: DataLoader, learner: LearnerClassification):
    """Trains trains a network using a learner, a given train_loader"""
    losses = []

    learner.train()

    for input, target in train_loader:
        loss = learner.learn(input.to(get_device()), target.to(get_device()))

        losses += [loss] * len(input)

    learner.step_schedule(epoch)
    return fmean(losses)


def evaluate(
    eval_loader: DataLoader, number_of_classes: int, learner: LearnerClassification
):
    with torch.no_grad():
        # Initialize all meters
        losses = []
        confusion = Confusion(number_of_classes)

        learner.eval()

        for input, target in eval_loader:
            # Prepare the inputs
            target = target.to(get_device())
            train_target, eval_target = (target, target)

            # Optimization
            loss, output = learner.forward_with_criterion(
                input.to(get_device()), train_target
            )

            losses += [loss] * len(input)

            # Update the performance meter
            confusion.add(output, eval_target)

    return fmean(losses), confusion.acc()


class Confusion:
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """

    def __init__(self, n_classes: int):
        self.k = n_classes
        self.conf = torch.LongTensor(n_classes, n_classes)
        self.conf.fill_(0)

    def add(self, output: torch.Tensor, target: torch.Tensor):
        if target.size(0) > 1:
            output = output.squeeze_()
            target = target.squeeze_()
        assert output.size(0) == target.size(0)
        if output.ndimension() > 1:  # it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(
                0
            ), "number of outputs does not match size of confusion matrix"

            _, pred = output.max(1)  # find the predicted class
        else:  # it is already the predicted class
            pred = output
        indices = (
            target * self.conf.stride(0) + pred.squeeze_().type_as(target)
        ).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        conf_flat = self.conf.view(-1)
        conf_flat.index_add_(0, indices, ones)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        return 0.0 if total == 0 else TP / total
