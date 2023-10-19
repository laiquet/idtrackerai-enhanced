import logging
from importlib import metadata

import torch
from torch.backends import mps


def get_device() -> torch.device:
    """Returns the current available device for PyTorch"""
    logging.debug("Using PyTroch %s", metadata.version("torch"))
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
