import logging
from importlib import metadata
from typing import Iterator, Protocol

import torch
from torch.backends import mps

from idtrackerai.utils import IdtrackeraiError, conf


class DataLoaderWithLabels(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]: ...


def get_device(user_device: str) -> torch.device:
    """Returns the current available device for PyTorch"""
    logging.debug("Using PyTorch %s", metadata.version("torch"))
    if user_device:
        try:
            device = torch.device(user_device)
        except RuntimeError as exc:
            raise IdtrackeraiError(
                f'Torch device name "{user_device}" not recognized.'
            ) from exc
        else:
            logging.info('Using user stated device "%s": %s', user_device, repr(device))
            return device
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


DEVICE = get_device(conf.DEVICE)
