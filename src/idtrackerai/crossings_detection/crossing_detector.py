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
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai import Blob, ListOfBlobs, Video
from idtrackerai.network import (
    LearnerClassification,
    NetworkParams,
    weights_xavier_init,
)
from idtrackerai.utils import conf

from .dataset.crossings_dataloader import get_training_data_loaders
from .dataset.crossings_dataset import get_train_validation_and_eval_blobs
from .model_area import ModelArea
from .network.predictor_crossing_detector import get_predictions_crossigns
from .network.stop_training_criteria_crossings import StopTraining
from .network.trainer_crossing_detector import train_deep_crossing


def _apply_area_and_unicity_heuristics(
    blobs_in_video: list[list[Blob]], number_of_animals: int, model_area: ModelArea
):
    """Applies `model_area` to every blob extracted from video

    Parameters
    ----------
    blobs_in_video : list[list[Blob]]
    number_of_animals : int
    model_area : ModelArea
    """
    logging.info(
        "Classifying Blobs as individuals or crossings "
        "depending on their area and the number of blobs in the frame"
    )
    for blobs_in_frame in blobs_in_video:
        unicity_cond = len(blobs_in_frame) == number_of_animals
        for blob in blobs_in_frame:
            blob.seems_like_individual = unicity_cond or model_area(blob.area)


def detect_crossings(list_of_blobs: ListOfBlobs, video: Video):
    """Classify all blobs in the video as being crossings or individuals.

    Parameters
    ----------
    list_of_blobs : <ListOfBlobs object>
        Collection of the Blob objects extracted from the video
    video :  <Video object>
        Object containing all the parameters of the video.
    model_area : function
        Model of the area of a single individual
    use_network : bool
        If True the Deep Crossing Detector is used to distinguish between
        individuals and crossings images. Otherwise only the model area is applied

    Returns
    -------

    trainer or list_of_blobs : TrainDeepCrossing or ListOfBlobs()
    """
    model_area = ModelArea(list_of_blobs, video.number_of_animals)

    _apply_area_and_unicity_heuristics(
        list_of_blobs.blobs_in_video, video.number_of_animals, model_area
    )

    train_blobs, val_blobs, eval_blobs = get_train_validation_and_eval_blobs(
        list_of_blobs.blobs_in_video, video.number_of_animals
    )

    if (
        len(train_blobs["crossings"])
        < conf.MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR
    ):
        logging.debug("There are not enough crossings to train the crossing detector")
        return
    logging.info("There are enough crossings to train the crossing detector")
    train_loader, val_loader = get_training_data_loaders(
        video.id_images_file_paths, train_blobs, val_blobs
    )
    logging.info("Setting crossing detector network parameters")
    network_params = NetworkParams(
        number_of_classes=2,
        architecture="DCD",
        save_folder=video.crossings_detector_folder,
        model_name="crossing_detector",
        image_size=video.id_image_size,
        use_gpu=True,
        optimizer="Adam",
        schedule=[30, 60],
        optim_args={"lr": conf.LEARNING_RATE_DCD},
        epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_DCD,
    )
    network_params.save()
    logging.info("Setting training criterion")
    criterion = CrossEntropyLoss(weight=torch.tensor(train_blobs["weights"]))
    crossing_detector_model = LearnerClassification.create_model(network_params)
    logging.info("Initialize networks params with Xavier initialization")
    crossing_detector_model.apply(weights_xavier_init)

    if network_params.use_gpu:
        torch.cuda.set_device(0)
        logging.info(
            'Sending model and criterion to GPU: "%s"', torch.cuda.get_device_name()
        )
        cudnn.benchmark = True  # make it train faster
        crossing_detector_model = crossing_detector_model.cuda()
        criterion = criterion.cuda()

    logging.info(f"Setting {network_params.optimizer} optimizer")
    if network_params.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            crossing_detector_model.parameters(), **network_params.optim_args
        )
    elif network_params.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            crossing_detector_model.parameters(), **network_params.optim_args
        )
    else:
        raise AttributeError(network_params.optimizer)

    logging.info("Setting scheduler")
    scheduler = MultiStepLR(optimizer, milestones=network_params.schedule, gamma=0.1)

    learner = LearnerClassification(
        crossing_detector_model, criterion, optimizer, scheduler
    )

    stop_training = StopTraining(network_params.epochs)

    model_diverged, best_model_path = train_deep_crossing(
        learner, train_loader, val_loader, network_params, stop_training
    )

    if model_diverged:
        logging.warning(
            "The model diverged. Falling back to individual-crossing discrimination by"
            " average area model."
        )
        return

    del train_loader
    del val_loader

    model_state = torch.load(best_model_path)
    crossing_detector_model.load_state_dict(model_state, strict=True)
    logging.info("Loaded best model weights from %s", best_model_path)

    logging.info("Using crossing detector to classify individuals and crossings")
    predictions = get_predictions_crossigns(
        video.id_images_file_paths,
        crossing_detector_model,
        eval_blobs,
        network_params.use_gpu,
    )

    logging.info(
        "Prediction results: %d individuals and %d crossings",
        predictions.count(0),
        predictions.count(1),
    )
    for blob, prediction in zip(eval_blobs, predictions):
        blob.is_an_individual = prediction != 1

    list_of_blobs.update_id_image_dataset_with_crossings(video.id_images_file_paths)
