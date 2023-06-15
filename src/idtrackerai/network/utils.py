import logging
from functools import cache

import torch
from torch.backends import mps


@cache
def get_device() -> torch.device:
    """Returns the current available device for PyTorch"""
    if torch.cuda.is_available():
        logging.info('Using Cuda backend with "%s"', torch.cuda.get_device_name(0))
        return torch.device(0)
    if mps.is_available():
        logging.info("Using MacOS Metal backend")
        return torch.device("mps")
    logging.warning(
        (
            "[bold red]No graphic device was found available[/], running neural"
            " networks on CPU. This may slow down the training steps."
        ),
        extra={"markup": True},
    )
    return torch.device("cpu")


def weights_xavier_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight.data)


def fc_weights_reinit(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


def normalize(tensor: torch.Tensor):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized Tensor image.
    """
    # TODO: This is kind of a batch normalization but not trained. Explore using real BN in idCNN.

    mean = torch.tensor([tensor.mean()])
    std = torch.tensor([tensor.std()])
    return tensor.sub_(mean[:, None, None]).div_(std[:, None, None])


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
            output = output.squeeze_()
            target = target.squeeze_()
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
        conf_flat = self.conf.view(-1)
        conf_flat.index_add_(0, indices, ones)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        return 0.0 if total == 0 else TP / total
