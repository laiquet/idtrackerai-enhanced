"""This file provides the template Learner. The Learner is used in training/evaluation loop
The Learner implements the training procedure for specific task.
The default Learner is from classification task."""

import logging
from pathlib import Path

import torch
from torch.nn import functional

from . import CNN, DEVICE, IdentificationModelBase, NetworkParams


class ClasificationCNN(IdentificationModelBase):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        softmax = functional.softmax(self.model.forward(images), dim=1)
        # https://github.com/pytorch/pytorch/issues/92311
        probabilities, pred = softmax.max(dim=1)
        return pred + 1, probabilities

    def load(self, path: Path | str):
        logging.info("Load model weights from %s", path)
        model_state: dict = torch.load(path)
        model_state.pop("val_acc", None)
        model_state.pop("test_acc", None)
        model_state.pop("ratio_accumulated", None)

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            logging.warning(
                "Loading a model from a version older than 5.1.7, "
                "going to translate the state dictionary."
            )
            translated_model_state = {
                "layers.0.weight": model_state["conv1.weight"],
                "layers.0.bias": model_state["conv1.bias"],
                "layers.3.weight": model_state["conv2.weight"],
                "layers.3.bias": model_state["conv2.bias"],
                "layers.6.weight": model_state["conv3.weight"],
                "layers.6.bias": model_state["conv3.bias"],
                "layers.9.weight": model_state["fc1.weight"],
                "layers.9.bias": model_state["fc1.bias"],
                "layers.11.weight": model_state["fc2.weight"],
                "layers.11.bias": model_state["fc2.bias"],
            }
            self.model.load_state_dict(translated_model_state, strict=True)


def load_CNN(
    learner_params: NetworkParams,
    knowledge_transfer: bool = False,
    device: torch.device = DEVICE,
) -> CNN:
    model = CNN(learner_params.image_size, learner_params.n_classes).to(device)
    if knowledge_transfer:
        model_path = learner_params.knowledge_transfer_model_file
        assert model_path is not None
    else:
        model_path = learner_params.load_model_path

    logging.info("Load model weights from %s", model_path)
    # The path to model file (*.best_model.pth). Do NOT use checkpoint file here
    model_state: dict = torch.load(model_path)
    model_state.pop("val_acc", None)
    model_state.pop("test_acc", None)
    model_state.pop("ratio_accumulated", None)

    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError:
        logging.warning(
            "Loading a model from a version older than 5.1.7, "
            "going to translate the state dictionary."
        )
        translated_model_state = {
            "layers.0.weight": model_state["conv1.weight"],
            "layers.0.bias": model_state["conv1.bias"],
            "layers.3.weight": model_state["conv2.weight"],
            "layers.3.bias": model_state["conv2.bias"],
            "layers.6.weight": model_state["conv3.weight"],
            "layers.6.bias": model_state["conv3.bias"],
            "layers.9.weight": model_state["fc1.weight"],
            "layers.9.bias": model_state["fc1.bias"],
            "layers.11.weight": model_state["fc2.weight"],
            "layers.11.bias": model_state["fc2.bias"],
        }
        model.load_state_dict(translated_model_state, strict=True)

    return model
