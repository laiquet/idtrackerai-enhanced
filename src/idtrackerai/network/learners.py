"""This file provides the template Learner. The Learner is used in training/evaluation loop
The Learner implements the training procedure for specific task.
The default Learner is from classification task."""
import logging
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from . import NetworkParams, models


class LearnerClassification:
    def __init__(
        self,
        model: Module,
        criterion: CrossEntropyLoss,
        optimizer: Optimizer,
        scheduler: MultiStepLR,
    ):
        super().__init__()
        logging.info("Setting the learner")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch: int = 0

    @staticmethod
    def create_model(learner_params: NetworkParams) -> Module:
        architecture = learner_params.architecture
        logging.info("Creating %s model", architecture)
        if architecture == "DCD":
            model = models.DCD
        elif architecture == "idCNN":
            model = models.idCNN
        elif architecture == "idCNN_adaptive":
            model = models.idCNN_adaptive
        else:
            raise ValueError(architecture)

        return model(
            out_dim=learner_params.n_classes, input_shape=learner_params.image_size
        )

    @classmethod
    def load_model(
        cls, learner_params: NetworkParams, knowledge_transfer: bool = False
    ):
        model = cls.create_model(learner_params)
        if knowledge_transfer:
            model_path = learner_params.knowledge_transfer_model_file
            assert model_path is not None
        else:
            model_path = learner_params.load_model_path

        logging.info("Load model weights from %s", model_path)
        # The path to model file (*.best_model.pth). Do NOT use checkpoint file here
        model_state: dict = torch.load(model_path)
        model_state.pop("val_acc", None)

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

    def learn(self, inputs, targets):
        loss, out = self.forward_with_criterion(inputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def step_schedule(self, epoch):
        self.epoch = epoch
        self.scheduler.step()

    def save_model(self, savename: Path, **extra_data):
        logging.info("Saving model at %s", savename)
        torch.save(self.model.state_dict() | extra_data, savename)
