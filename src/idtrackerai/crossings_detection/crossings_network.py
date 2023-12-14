from pathlib import Path

import numpy as np
import torch

from idtrackerai import Blob
from idtrackerai.network import CNN, DEVICE
from idtrackerai.utils import track

from .crossings_dataset import get_crossing_dataloader


def get_predictions_crossigns(
    id_images_file_paths: list[Path], model: CNN, blobs: list[Blob]
):
    loader = get_crossing_dataloader(id_images_file_paths, blobs, "test")

    model.eval()
    predictions = np.empty(len(blobs), np.int32)
    index = 0
    with torch.no_grad():
        for input, _target in track(loader, "Predicting crossings"):
            # Inference
            output = model.forward(input.to(DEVICE))
            # https://github.com/pytorch/pytorch/issues/92311
            pred = output.max(dim=1).indices

            predictions[index : index + len(pred)] = pred.cpu()
            index += len(pred)
    assert index == len(predictions)
    return predictions
