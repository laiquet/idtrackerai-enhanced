from statistics import fmean

import torch
from torch.utils.data import DataLoader

from . import DEVICE, LearnerClassification


def train(epoch: int, train_loader: DataLoader, learner: LearnerClassification):
    """Trains trains a network using a learner, a given train_loader"""
    losses = []

    learner.train()

    for input, target in train_loader:
        loss = learner.learn(input.to(DEVICE), target.to(DEVICE))

        losses += [loss] * len(input)

    learner.step_schedule(epoch)
    return fmean(losses)


def evaluate(eval_loader: DataLoader, n_classes: int, learner: LearnerClassification):
    with torch.no_grad():
        # Initialize all meters
        losses = []
        confusion = Confusion(n_classes)

        learner.eval()

        for input, target in eval_loader:
            # Prepare the inputs
            target = target.to(DEVICE)

            # Optimization
            loss, output = learner.forward_with_criterion(input.to(DEVICE), target)

            losses += [loss] * len(input)

            # Update the performance meter
            confusion.add(output, target)

    return fmean(losses), confusion.acc()


class Confusion:
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """

    def __init__(self, n_classes: int):
        self.k = n_classes
        self.conf = torch.LongTensor(n_classes, n_classes)
        self.conf.fill_(0)

    def add(self, output: torch.Tensor, target: torch.Tensor):
        if target.size(0) > 1:
            output = output.squeeze()
            target = target.squeeze()
        assert output.size(0) == target.size(0)
        if output.ndimension() > 1:  # it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(
                0
            ), "number of outputs does not match size of confusion matrix"

            _, pred = output.max(1)  # find the predicted class
        else:  # it is already the predicted class
            pred = output
        indices = (
            target * self.conf.stride(0) + pred.squeeze_().type_as(target)
        ).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self.conf.view(-1).index_add_(0, indices, ones)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        return 0.0 if total == 0 else TP / total
