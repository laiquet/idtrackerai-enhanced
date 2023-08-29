import logging

import torch
from torch.backends import mps


def get_device() -> torch.device:
    """Returns the current available device for PyTorch"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info('Using Cuda backend with "%s"', torch.cuda.get_device_name(device))
        return device
    if mps.is_available():
        logging.info("Using MacOS Metal backend")
        return torch.device("mps")
    logging.warning(
        "[bold red]No graphic device was found available[/], running neural"
        " networks on CPU. This may slow down the training steps.",
        extra={"markup": True},
    )
    return torch.device("cpu")


DEVICE = get_device()


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
