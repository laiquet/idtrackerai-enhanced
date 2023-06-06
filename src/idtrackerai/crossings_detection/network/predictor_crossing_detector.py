from pathlib import Path

import torch
from torch.nn import Module

from idtrackerai import Blob
from idtrackerai.crossings_detection.dataset.crossings_dataloader import (
    get_test_data_loader,
)
from idtrackerai.network import get_device
from idtrackerai.utils import track


def get_predictions_crossigns(
    id_images_file_paths: list[Path], model: Module, blobs: list[Blob]
):
    loader = get_test_data_loader(id_images_file_paths, blobs)
    predictions = []

    model.eval()
    for input, _target in track(loader, "Predicting crossings"):
        # Prepare the inputs

        with torch.no_grad():
            input = input.to(get_device())

        # Inference
        output = model(input)
        pred = output.argmax(1)  # find the predicted class

        predictions.extend(pred.cpu().numpy())

    return predictions
