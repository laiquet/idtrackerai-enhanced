import logging
import sys
from contextlib import suppress

import numpy as np
import torch
from rich.console import Console
from rich.status import Status
from torch.backends import cudnn
from torch.utils.data import DataLoader

from idtrackerai.network import (
    LearnerClassification,
    NetworkParams,
    evaluate,
    get_device,
    train,
)
from idtrackerai.utils import CustomError, conf, track

from .identity_dataset import get_test_data_loader


class StopTraining:
    """Stops the training of the network according to the conditions specified
    in :meth:`__call__`

    Attributes
    ----------
    num_epochs : int
        Number of epochs before starting the training
    number_of_animals : int
        Number of animals in the video
    epochs_before_checking_stopping_conditions : int
        Number of epochs before starting to check the stopping conditions
    overfitting_counter : int
        Counts the number of consecutive overfitting epochs during training
    first_accumulation_flag : bool
        Flag to indicate that it is the first step of the accumulation

    """

    epochs_before_checking_stopping_conditions = 10

    def __init__(self, number_of_animals: int, is_first_accumulation: bool = False):
        logging.info("Setting the stopping criteria", stacklevel=3)
        self.num_epochs = conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN
        self.number_of_animals = number_of_animals
        # number of epochs in which the network is overfitting
        # before stopping the training
        self.overfitting_counter = 0
        self.is_first_accumulation = is_first_accumulation
        self.epochs_completed = -1

    def __call__(
        self,
        loss_training: float,
        loss_validation: list,
        accuracy_validation: float,
        status: Status,
    ):
        """Returns True when one of the conditions to stop the training is
        satisfied, otherwise it returns False"""
        self.epochs_completed += 1

        if self.epochs_completed > 0 and (
            np.isnan(loss_training) or np.isnan(loss_validation[-1])
        ):
            status.stop()
            logging.error(
                "The model diverged. Oops. Check the hyperparameters and "
                "the architecture of the network."
            )
            return True
        # check if it did not reached the epochs limit
        if self.epochs_completed >= self.num_epochs:
            status.stop()
            logging.warning(
                "The number of epochs completed is larger than the number "
                "of epochs set for training, we stop the training"
            )
            return True

        if self.epochs_completed <= self.epochs_before_checking_stopping_conditions:
            return False

        # check that the model is not overfitting or if it reached
        # a stable saddle (minimum)
        current_loss = loss_validation[-1]
        previous_loss = np.nanmean(
            loss_validation[-self.epochs_before_checking_stopping_conditions : -1]
        )

        # The validation loss in the first 10 epochs could have exploded
        # but being decreasing.
        if np.isnan(previous_loss):
            previous_loss = sys.float_info[0]
        losses_difference = previous_loss - current_loss

        # check overfitting
        if losses_difference < 0.0:
            self.overfitting_counter += 1
            if (
                not self.is_first_accumulation
                and self.overfitting_counter >= conf.OVERFITTING_COUNTER_THRESHOLD_IDCNN
            ):
                status.stop()
                logging.info("Overfitting")
                return True
            if (
                self.is_first_accumulation
                and self.overfitting_counter
                > conf.OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM
            ):
                # logging.info(f"Overfitting counter, {self.overfitting_counter}")
                status.stop()
                logging.info("Overfitting first accumulation")
                return True
        else:
            self.overfitting_counter = 0

        # check if the error is not decreasing much

        if self.is_first_accumulation and np.abs(
            losses_difference
        ) < conf.LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN * 10 ** (
            int(np.log10(current_loss)) - 1
        ):
            status.stop()
            logging.info("The losses difference is very small, we stop the training")
            return True
        if np.abs(
            losses_difference
        ) < conf.LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN * 10 ** (
            int(np.log10(current_loss)) - 1
        ):
            status.stop()
            logging.info("The losses difference is very small, we stop the training")
            return True

        # if the individual accuracies in validation are 1.
        # for all the animals
        if accuracy_validation == 1.0:
            status.stop()
            logging.info(
                "The individual accuracies in validation is 100% for "
                "all the individuals, we stop the training"
            )
            return True

        # if the validation loss is 0.
        if previous_loss == 0.0 or current_loss == 0.0:
            status.stop()
            logging.info("The validation loss is 0.0, we stop the training")
            return True

        return False


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

            train_loss = train(epoch, train_loader, learner)
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


def get_predictions_identities(
    model: torch.nn.Module, images: np.ndarray, network_params: NetworkParams
):
    logging.debug("Generating prediction data set with %d images", len(images))
    loader = get_test_data_loader({"images": images}, network_params.number_of_classes)
    predictions = []
    softmax_probs = []

    logging.debug("Using trained network to predict images identities")
    if not next(model.parameters()).is_cuda:
        logging.info("Sending model and criterion to GPU")
        cudnn.benchmark = True  # make it train faster
        model = model.to(get_device())

    model.eval()
    for input_, _target in track(loader, "Predicting identities"):
        # Prepare the inputs
        with torch.no_grad():
            input_ = input_.to(get_device())

        # Inference
        with torch.no_grad():
            softmax = model.softmax_probs(input_)  # type: ignore
            pred = softmax.argmax(1)  # find the predicted class

            predictions += pred.tolist()
            softmax_probs += softmax.tolist()

    return np.asarray(predictions) + 1, np.asarray(softmax_probs)
