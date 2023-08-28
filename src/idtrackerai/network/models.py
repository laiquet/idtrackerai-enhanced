from itertools import pairwise
from typing import Sequence

from torch import Tensor, nn


class DCD(nn.Module):
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

    def forward(self, x: Tensor):
        return self.layers(x)


class idCNN(nn.Module):
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

    def forward(self, x: Tensor):
        return self.layers(x)


class idCNN_adaptive(nn.Module):
    def __init__(self, input_shape: Sequence[int], out_dim):
        """
        input_shape: tuple (width, height, channels)
        out_dim: int
        """
        super().__init__()

        self.out_dim = out_dim
        num_channels = [input_shape[-1], 16, 64, 100]

        kernel_size = 5
        self.width_adaptive_pool = 3

        # Convolutional and pooling layers
        cnn_layers = []
        for i, (num_ch_in, num_ch_out) in enumerate(pairwise(num_channels)):
            if i > 0:
                # no pooling after input
                cnn_layers.append(nn.MaxPool2d(2, stride=2))

            cnn_layers.append(nn.Conv2d(num_ch_in, num_ch_out, kernel_size, padding=2))
            cnn_layers.append(nn.ReLU(inplace=True))

        cnn_layers.append(nn.AdaptiveAvgPool2d(self.width_adaptive_pool))
        self.layers = nn.Sequential(
            *cnn_layers,
            nn.Flatten(),
            nn.Linear(num_channels[-1] * self.width_adaptive_pool**2, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_dim),
        )

    def forward(self, x: Tensor):
        return self.layers(x)
