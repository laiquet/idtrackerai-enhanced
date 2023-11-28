from contextlib import suppress
from typing import Sequence

from torch import Tensor, nn


class CNN(nn.Module):
    def __init__(self, input_shape: Sequence[int], out_dim: int):
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

    def forward(self, x: Tensor) -> Tensor:
        # per image normalization
        x -= x.mean((1, 2, 3), keepdim=True)
        with suppress(ValueError):
            x /= x.std((1, 2, 3), keepdim=True)

        return self.layers(x)
