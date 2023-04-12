"""This file provides the template Learner. The Learner is used in training/evaluation loop
The Learner implements the training procedure for specific task.
The default Learner is from classification task."""
import logging
from pathlib import Path

import torch
from torch import nn

from . import NetworkParams, models


class LearnerClassification(nn.Module):
    def __init__(self, model: nn.Module, criterion, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.model_path = None

    @staticmethod
    def create_model(learner_params: NetworkParams) -> nn.Module:
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
    def load_model(cls, learner_params: NetworkParams, scope=""):
        model = cls.create_model(learner_params)
        if scope == "knowledge_transfer":
            model_path = learner_params.knowledge_transfer_model_file
        else:
            model_path = learner_params.load_model_path
        assert model_path is not None

        logging.info("Load model weights from %s", model_path)
        # The path to model file (*.best_model.pth). Do NOT use checkpoint file here
        # model_state = torch.load(
        #     model_path, map_location=lambda storage, loc: storage
        # )  # Load to CPU as the default!
        model_state = torch.load(model_path)
        # The pretrained state dict doesn't need to fit the model
        model.load_state_dict(model_state, strict=True)
        return model

    def forward(self, x):
        return self.model.forward(x)

    def forward_with_criterion(self, inputs, targets, **kwargs):
        out = self.forward(inputs)
        targets = targets.long()
        return self.criterion(out, targets), out

    def learn(self, inputs, targets, **kwargs):
        with torch.autograd.set_detect_anomaly(True):
            loss, out = self.forward_with_criterion(inputs, targets, **kwargs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, out

    def step_schedule(self, epoch):
        self.epoch = epoch
        self.scheduler.step()
        # for param_group in self.optimizer.param_groups:
        # print("LR:", param_group["lr"])

    def save_model(self, savename: Path):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.model_path = savename.parent / (savename.name + ".pth")
        torch.save(model_state, self.model_path)

    def snapshot(self, savename: Path) -> Path:
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        checkpoint = {
            "epoch": self.epoch,
            "model": model_state,
            "optimizer": optim_state,
        }
        torch.save(checkpoint, savename.parent / (savename.name + ".checkpoint.pth"))
        self.save_model(savename.parent / (savename.name + ".model"))
        assert self.model_path is not None
        return self.model_path
