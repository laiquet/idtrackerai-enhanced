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

import torch
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import GlobalFragment, ListOfFragments
from idtrackerai.network import (
    LearnerClassification,
    NetworkParams,
    fc_weights_reinit,
    get_device,
)
from idtrackerai.utils import conf

from .identity_dataset import get_training_data_loaders, split_data_train_and_validation
from .identity_network import StopTraining, TrainIdentification


def pre_train_global_fragment(
    number_of_animals: int,
    accumulation_step: int,
    identification_model: Module,
    network_params: NetworkParams,
    pretraining_global_fragment: GlobalFragment,
    list_of_fragments: ListOfFragments,
):
    """Performs pretraining on a single global fragments

    Parameters
    ----------
    net : <ConvNetwork obejct>
        an instance of the class :class:`~idCNN.ConvNetwork`
    pretraining_global_fragment : <GlobalFragment object>
        an instance of the class :class:`~globalfragment.GlobalFragment`
    list_of_fragments : <ListOfFragments object>
        an instance of the class :class:`~list_of_fragments.ListOfFragments`
    global_epoch : int
        global counter of the training epoch in pretraining
    store_accuracy_and_error : bool
        if True the values of the loss function, accuracy and individual
        accuracy will be stored
    save_summaries : bool
        if True tensorflow summaries will be generated and stored to allow
        tensorboard visualisation of both loss and activity histograms
    store_training_accuracy_and_loss_data : <Store_Accuracy_and_Loss object>
        an instance of the class :class:`~Store_Accuracy_and_Loss`
    store_validation_accuracy_and_loss_data : <Store_Accuracy_and_Loss object>
        an instance of the class :class:`~Store_Accuracy_and_Loss`

    Returns
    -------
    <ConvNetwork object>
        network with updated parameters after training
    float
        ration of images used for pretraining over the total number of
        available images
    int
        global epoch counter updated after the training session
    <Store_Accuracy_and_Loss object>
        updated with the values collected on the training set of labelled
        images
    <Store_Accuracy_and_Loss object>
        updated with the values collected on the validation set of labelled
        images
    <ListOfFragments objects>
        list of instances of the class :class:`~fragment.Fragment`
    """
    # Get images and labels from the current global fragment
    images, labels = pretraining_global_fragment.get_images_and_labels(
        list_of_fragments.id_images_file_paths
    )

    train_data, val_data = split_data_train_and_validation(
        images, labels, validation_proportion=conf.VALIDATION_PROPORTION
    )
    logging.debug(f"images: {images.shape} {images.dtype}")
    logging.debug(f"labels: {labels.shape}")

    # Set data loaders
    train_loader, val_loader = get_training_data_loaders(
        number_of_animals, train_data, val_data
    )

    # Set criterion
    logging.info("Setting training criterion")
    criterion = CrossEntropyLoss(weight=torch.tensor(train_data["weights"]))

    # Re-initialize fully-connected layers
    identification_model.apply(fc_weights_reinit)

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
        network_params.number_of_classes, is_first_accumulation=accumulation_step == 0
    )

    TrainIdentification(
        learner, train_loader, val_loader, network_params, stop_training
    )

    for fragment in pretraining_global_fragment.individual_fragments:
        fragment.used_for_pretraining = True

    ratio_of_pretrained_images = (
        list_of_fragments.compute_ratio_of_images_used_for_pretraining()
    )
    logging.debug(
        "ratio of images used during pretraining: "
        f"{ratio_of_pretrained_images:.2%} (if higher than "
        f"{conf.MAX_RATIO_OF_PRETRAINED_IMAGES:.2%} we stop pretraining)"
    )

    return (
        identification_model,
        ratio_of_pretrained_images,
        list_of_fragments,
        network_params.model_path,
    )
