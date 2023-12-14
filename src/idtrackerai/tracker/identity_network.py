import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional

from idtrackerai.network import CNN, DEVICE
from idtrackerai.utils import track

from .identity_dataset import get_onthefly_dataloader


def get_predictions_identities(
    model: CNN, image_location: Sequence[tuple[int, int]], id_images_paths: list[Path]
):
    logging.debug(
        "Predicting identities of %d images", len(image_location), stacklevel=3
    )
    predictions = np.empty(len(image_location), np.int32)
    max_softmax = np.empty(len(image_location), np.float32)
    index = 0
    model.eval()
    dataloader = get_onthefly_dataloader(image_location, id_images_paths)
    with torch.no_grad():
        for images, _labels in track(dataloader, "Predicting identities"):
            softmax = functional.softmax(model.forward(images.to(DEVICE)), dim=1)
            # https://github.com/pytorch/pytorch/issues/92311
            maximum, pred = softmax.max(dim=1)

            predictions[index : index + len(pred)] = (pred + 1).cpu()
            max_softmax[index : index + len(pred)] = maximum.cpu()
            index += len(pred)
    assert index == len(predictions) == len(max_softmax)
    return predictions, max_softmax
