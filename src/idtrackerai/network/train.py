import torch

from . import CNN, DEVICE, DataLoaderWithLabels, LearnerClassification


def train(
    epoch: int, train_loader: DataLoaderWithLabels, learner: LearnerClassification
):
    """Trains trains a network using a learner, a given train_loader"""
    losses = 0
    n_predictions = 0

    learner.train()

    for input, target in train_loader:
        loss = learner.learn(input.to(DEVICE), target.to(DEVICE))

        losses += loss.item() * len(input)
        n_predictions += len(input)

    learner.step_schedule(epoch)
    return losses / n_predictions


def evaluate(eval_loader: DataLoaderWithLabels, learner: LearnerClassification):
    with torch.no_grad():
        losses = 0
        n_predictions = 0
        n_right_guess = 0

        learner.eval()

        for input, target in eval_loader:
            target = target.to(DEVICE)

            loss, output = learner.forward_with_criterion(input.to(DEVICE), target)
            n_predictions += len(target)
            n_right_guess += (output.max(1).indices == target).count_nonzero().item()

            losses += loss.item() * len(input)

    return losses / n_predictions, n_right_guess / n_predictions


def evaluate_only_acc(eval_loader: DataLoaderWithLabels, model: CNN):
    with torch.no_grad():
        model.eval()
        n_predictions = 0
        n_right_guess = 0

        for input, target in eval_loader:
            predictions = model.forward(input.to(DEVICE)).max(1).indices
            n_predictions += len(target)
            n_right_guess += (predictions == target.to(DEVICE)).count_nonzero().item()

    return n_right_guess / n_predictions
