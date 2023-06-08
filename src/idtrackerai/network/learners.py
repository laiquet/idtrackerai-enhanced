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


class LearnerClassification(Module):
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
        logging.info("Creating model")
        if learner_params.architecture == "DCD":
            model = models.DCD
        elif learner_params.architecture == "idCNN":
            model = models.idCNN
        elif learner_params.architecture == "idCNN_adaptive":
            model = models.idCNN_adaptive
        else:
            raise ValueError(learner_params.architecture)

        return model(
            out_dim=learner_params.number_of_classes,
            input_shape=learner_params.image_size,
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
        # model_state = torch.load(
        #     model_path, map_location=lambda storage, loc: storage
        # )  # Load to CPU as the default!
        model_state: dict = torch.load(model_path)
        model_state.pop("val_acc", None)
        # The pretrained state dict doesn't need to fit the model
        model.load_state_dict(model_state, strict=True)
        return model

    def forward(self, x):
        return self.model.forward(x)

    def forward_with_criterion(self, inputs, targets):
        out = self.forward(inputs)
        targets = targets.long()
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
