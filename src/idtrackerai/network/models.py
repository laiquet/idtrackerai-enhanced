from torch import nn


def compute_output_width(width, kernel, padding, stride):
    return int(((width - kernel + 2 * padding) / stride) + 1)


class DCD(nn.Module):
    def __init__(self, input_shape, out_dim):
        """
        input_shape: tuple (channels, width, height)
        out_dim: int
        """
        super().__init__()

        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(
            input_shape[-1],
            16,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(input_shape[1], 5, 2, 1)
        self.pool1 = nn.MaxPool2d(
            2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv2 = nn.Conv2d(
            16,
            64,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(w, 5, 2, 1)
        self.pool2 = nn.MaxPool2d(
            2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv3 = nn.Conv2d(
            64,
            100,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.w = compute_output_width(w, 5, 2, 1)
        self.fc1 = nn.Linear(100 * w * w, 100)
        self.fc2 = nn.Linear(100, out_dim)

        self.conv = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.pool1,
            self.conv2,
            nn.ReLU(inplace=True),
            self.pool2,
            self.conv3,
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(self.fc1, nn.ReLU(inplace=True))

        self.last = self.fc2

    def features(self, x):
        x = self.conv(x)
        return self.linear(x.view(-1, 100 * self.w * self.w))

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        return self.logits(self.features(x))


class idCNN(nn.Module):
    def __init__(self, input_shape, out_dim):
        """
        input_shape: tuple (channels, width, height)
        out_dim: int
        """
        super().__init__()

        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(
            input_shape[-1],
            16,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(input_shape[1], 5, 2, 1)
        self.pool1 = nn.MaxPool2d(
            2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv2 = nn.Conv2d(
            16,
            64,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        w = compute_output_width(w, 5, 2, 1)
        self.pool2 = nn.MaxPool2d(
            2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )
        w = compute_output_width(w, 2, 0, 2)
        self.conv3 = nn.Conv2d(
            64,
            100,
            5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.w = compute_output_width(w, 5, 2, 1)
        self.fc1 = nn.Linear(100 * w * w, 100)
        self.fc2 = nn.Linear(100, out_dim)

        self.conv = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.pool1,
            self.conv2,
            nn.ReLU(inplace=True),
            self.pool2,
            self.conv3,
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(self.fc1, nn.ReLU(inplace=True))

        self.last = self.fc2

        self.softmax = nn.Softmax(dim=1)

    def features(self, x):
        x = self.conv(x)
        return self.linear(x.view(-1, 100 * self.w * self.w))

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        return self.logits(self.features(x))

    def softmax_probs(self, x):
        return self.softmax(self.forward(x))


class idCNN_adaptive(nn.Module):
    def __init__(self, input_shape, out_dim):
        """
        input_shape: tuple (width, height, channels)
        out_dim: int
        """
        super().__init__()

        self.out_dim = out_dim
        num_channels = [input_shape[-1], 16, 64, 100]
        cnn_kwargs = {
            "stride": 1,
            "padding": 2,
            "dilation": 1,
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros",
        }
        maxpool_kwargs = {
            "stride": 2,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        }
        kernel_size = 5
        self.width_adaptive_pool = 3

        # Convolutional and pooling layers
        cnn_layers = []
        for i, (num_ch_in, num_ch_out) in enumerate(
            zip(num_channels[:-1], num_channels[1:])
        ):
            if i > 0:
                # no pooling after input
                cnn_layers.append(nn.MaxPool2d(2, **maxpool_kwargs))

            cnn_layers.append(
                nn.Conv2d(num_ch_in, num_ch_out, kernel_size, **cnn_kwargs)
            )
            cnn_layers.append(nn.ReLU(inplace=True))

        cnn_layers.append(nn.AdaptiveAvgPool2d(self.width_adaptive_pool))
        self.conv = nn.Sequential(*cnn_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(num_channels[-1] * self.width_adaptive_pool**2, 100)
        self.fc2 = nn.Linear(100, out_dim)
        self.linear = nn.Sequential(self.fc1, nn.ReLU(inplace=True))
        self.last = self.fc2

        self.softmax = nn.Softmax(dim=1)

    def features(self, x):
        x = self.conv(x)
        return self.linear(x.view(-1, 100 * self.width_adaptive_pool**2))

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        return self.logits(self.features(x))

    def softmax_probs(self, x):
        return self.softmax(self.forward(x))
