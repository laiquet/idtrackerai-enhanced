"""This file provides the template Learner. The Learner is used in training/evaluation loop
The Learner implements the training procedure for specific task.
The default Learner is from classification task."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from . import CNN, DEVICE, NetworkParams


@dataclass(slots=True)
class LearnerClassification:
    model: CNN
    criterion: CrossEntropyLoss
    optimizer: Optimizer
    scheduler: LRScheduler | None = None

    @classmethod
    def load_model(
        cls,
        learner_params: NetworkParams,
        knowledge_transfer: bool = False,
        device: torch.device = DEVICE,
    ) -> CNN:
        model = CNN.from_network_params(learner_params).to(device)
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

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward_with_criterion(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model.forward(inputs)
        return self.criterion(out, targets), out

    def learn(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss, out = self.forward_with_criterion(inputs, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss

    def step_schedule(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def save_model(self, savename: Path, **extra_data):
        logging.info("Saving model at %s", savename)
        torch.save(self.model.state_dict() | extra_data, savename)
