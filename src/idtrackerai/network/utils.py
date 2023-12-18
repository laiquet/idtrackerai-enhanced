import logging
from importlib import metadata
from typing import Iterator, Protocol

import torch
from torch.backends import mps


class DataLoaderWithLabels(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]: ...


def get_device() -> torch.device:
    """Returns the current available device for PyTorch"""
    logging.debug("Using PyTorch %s", metadata.version("torch"))
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
