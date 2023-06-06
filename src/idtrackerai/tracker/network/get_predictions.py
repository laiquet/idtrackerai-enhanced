import logging

import numpy as np
import torch
from torch.backends import cudnn

from idtrackerai.network import NetworkParams, get_device
from idtrackerai.tracker.dataset.identification_dataloader import get_test_data_loader
from idtrackerai.utils import track


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

            predictions.extend(pred.cpu().numpy())
            softmax_probs.extend(softmax.cpu().numpy())

    return np.asarray(predictions) + 1, np.asarray(softmax_probs)
