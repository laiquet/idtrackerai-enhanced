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

import numpy as np
import torch

from .utils import Confusion, prepare_task_target


def train(epoch, train_loader, learner, network_params):
    """Trains trains a network using a learner, a given train_loader and a set of network_params

    :param epoch: current epoch
    :param train_loader: dataloader
    :param learner: learner from learner.py
    :param network_params: networks params from networks_params.py
    :return: losses (tuple) and accuracy
    """

    # Initialize all meters
    losses = []
    if network_params.loss in ("CEMCL", "CEMCL_weighted"):
        losses_CE = []
        losses_MCL = []
    confusion = Confusion(network_params.number_of_classes)

    # Setup learner's configuration
    learner.train()

    # The optimization loop
    for input_, target in train_loader:
        # mask
        mask = None
        if network_params.apply_mask:
            mask = torch.from_numpy(~np.eye(len(target), dtype=bool))
        # Prepare the inputs
        if network_params.use_gpu:
            input_ = input_.cuda()
            target = target.cuda()
            if mask is not None:
                mask = mask.cuda()
        train_target, eval_target = prepare_task_target(
            target, network_params, mask=mask
        )

        # Optimization
        if "weighted" in network_params.loss:
            loss, output = learner.learn(
                input_, train_target, w_MCL=network_params.w_MCL, mask=mask
            )
        else:
            loss, output = learner.learn(input_, train_target, mask=mask)

        with torch.no_grad():
            # Update the performance meter
            if network_params.loss in ("CEMCL", "CEMCL_weighted"):
                confusion.add(output[0], eval_target)
            else:
                confusion.add(output, eval_target)

        # Mini-Logs
        losses += [loss] * input_.size(0)
        if network_params.loss in ("CEMCL", "CEMCL_weighted"):
            losses_CE += [output[1]] * input_.size(0)
            losses_MCL += [output[2]] * input_.size(0)

    learner.step_schedule(epoch)
    # print loss avg
    # print(losses.avg)
    # Loss-specific information
    if network_params.loss == "CE":
        pass
        # print("[Train] ACC: ", confusion.acc())
    elif network_params.loss in ("MCL", "CEMCL", "CEMCL_weighted"):
        network_params.cluster2Class = tuple(
            confusion.optimal_assignment(train_loader.num_classes)
        )  # Save the mapping in network_params to use in eval
        # print(network_params.cluster2Class)
        if network_params.out_dim <= 20:  # Avoid to print a large confusion matrix
            confusion.show()
        # print("Clustering scores:", confusion.clusterscores())
        # print("[Train] ACC: ", confusion.acc())

    if network_params.loss in ("CEMCL", "CEMCL_weighted"):
        return (fmean(losses), fmean(losses_CE), fmean(losses_MCL)), confusion.acc()
    return (fmean(losses), None, None), confusion.acc()
