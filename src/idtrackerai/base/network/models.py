import logging
from abc import ABC
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass
class IdentificationModelBase(ABC):
    model: nn.Module

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        "Takes a tensor images of size (Batch size, 1 ,Height, Width) in the range [0,1] and outputs another tensor of size (B, n_animals) for the predicted identities in the range [1, n_animals]"
        raise NotImplementedError

    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(images)

    def load(self, path: Path) -> None:
        raise NotImplementedError

    def save(self, path: Path, **extra_data) -> None:
        assert not path.is_dir()
        path = path.with_suffix(".pt")
        logging.info("Saving %s at %s", self.__class__.__name__, path)
        torch.save(self.model.state_dict() | extra_data, path)


class CNN(nn.Module):
    def __init__(self, input_shape: Sequence[int], out_dim: int):
        logging.info("Creating CNN model")
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_shape[-1], 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 100, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(100 * (input_shape[1] // 4) ** 2, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_dim),
        )

        self.reinitilaize()

    def forward(self, x: Tensor) -> Tensor:
        # per image normalization
        x -= x.mean((1, 2, 3), keepdim=True)
        with suppress(ValueError):
            x /= x.std((1, 2, 3), keepdim=True)

        return self.layers(x)

    def reinitilaize(self):
        logging.info("Reinitializing model")

        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(init_func)

    def fully_connected_reinitialization(self):
        logging.info("Reinitializing only fully connected layers")

        def init_func(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(init_func)
