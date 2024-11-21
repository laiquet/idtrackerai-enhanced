import logging
import multiprocessing
from importlib import metadata

import torch
from torch.backends import mps

from idtrackerai import IdtrackeraiError, conf


def _get_device(user_device: str) -> torch.device:
    """Returns the current available device for PyTorch"""
    logging.debug("Using PyTorch %s", metadata.version("torch"))
    if user_device:
        try:
            device = torch.device(user_device)
        except RuntimeError as exc:
            raise IdtrackeraiError(
                f"Torch device name {user_device!r} not recognized."
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

    # Only the main process raises the Warning, spawned processes don't need to insist
    logging.log(
        (
            logging.WARNING
            if multiprocessing.current_process().name == "MainProcess"
            else logging.INFO
        ),
        "[bold red]No graphic device was found available[/], running neural"
        " networks on CPU. This may slow down the training steps.",
        extra={"markup": True},
    )
    return torch.device("cpu")


DEVICE = _get_device(conf.DEVICE)
