import logging
from abc import ABC
from contextlib import suppress
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18(ResNet):
    def __init__(self, n_channels_in: int = 1, n_dimensions_out: int = 8) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=n_dimensions_out)
        if n_channels_in != 3:
            # adapt first conv layer to our single channel images (not RGB)
            self.conv1 = torch.nn.Conv2d(
                n_channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    @classmethod
    def from_file(cls, path: Path):
        assert path.is_file()
        model_state_dict = torch.load(path)
        n_dimensions_out = len(model_state_dict["fc.weight"])
        n_channels_in = model_state_dict["conv1.weight"].shape[1]
        model = cls(n_channels_in, n_dimensions_out)
        model.load_state_dict(model_state_dict)
        return model


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

    @classmethod
    def load(cls, image_size: Sequence[int], model_path: Path):
        assert model_path.is_dir()

        for name in (  # v5.0.0 compatibility
            "identifier_cnn.model.pt",
            "identification_network.model.pth",
            "identification_network_.model.pth",
            "supervised_identification_network.model.pth",
            "supervised_identification_network_.model.pth",
        ):
            if (model_path / name).is_file():
                model_path = model_path / name
                break
        else:
            raise FileNotFoundError(model_path)

        logging.info("Load model weights from %s", model_path)
        # The path to model file (*.best_model.pth). Do NOT use checkpoint file here
        model_state: dict = torch.load(model_path)
        model_state.pop("val_acc", None)
        model_state.pop("test_acc", None)
        model_state.pop("ratio_accumulated", None)

        if "fc2.weight" in model_state:
            n_classes = len(model_state["fc2.weight"])
        else:
            n_classes = len(model_state["layers.11.weight"])

        model = cls(image_size, n_classes)
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            logging.warning(
                "Loading a model from a version older than 5.1.7, "
                "going to translate the state dictionary."
            )
            translated_model_state = {
                "layers.0.weight": model_state["conv1.weight"],
                "layers.0.bias": model_state["conv1.bias"],
                "layers.3.weight": model_state["conv2.weight"],
                "layers.3.bias": model_state["conv2.bias"],
                "layers.6.weight": model_state["conv3.weight"],
                "layers.6.bias": model_state["conv3.bias"],
                "layers.9.weight": model_state["fc1.weight"],
                "layers.9.bias": model_state["fc1.bias"],
                "layers.11.weight": model_state["fc2.weight"],
                "layers.11.bias": model_state["fc2.bias"],
            }
            model.load_state_dict(translated_model_state, strict=True)

        return model


class IdentifierBase(ABC):
    model: nn.Module
    model_weights_filename: str

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        "Takes a tensor images of size (Batch size, 1 ,Height, Width) in the range [0,1] and outputs another tensor of size (Batch size, n_animals) for the predicted identities in the range [1, n_animals]"
        raise NotImplementedError

    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(images)

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError

    def save(self, path: Path, **extra_data) -> None:
        assert path.is_dir()
        path = path / self.model_weights_filename
        logging.info("Saving %s at %s", self.__class__.__name__, path)
        torch.save(self.model.state_dict() | extra_data, path)


class IdentifierCNN(IdentifierBase):
    model: CNN  # pyright: ignore[reportIncompatibleVariableOverride] CNN is subclass of torch.nn.Module
    model_weights_filename: str = "identifier_cnn.model.pt"

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        softmax = nn.functional.softmax(self.model.forward(images), dim=1)
        # https://github.com/pytorch/pytorch/issues/92311
        probabilities, pred = softmax.max(dim=1)
        return pred + 1, probabilities

    def save(self, path: Path, **extra_data) -> None:
        assert path.is_dir()
        return super().save(path, **extra_data)

    @classmethod
    def load(cls, image_size: Sequence[int], model_path: Path):
        return cls(CNN.load(image_size, model_path))


class IdentifierContrastive(IdentifierBase):
    cluster_centers: Tensor
    model_weights_filename: str = "identifier_contrastive.model.pt"
    cluster_centers_filename: str = "identifier_contrastive.cluster_centers.csv"

    def __init__(self, model: nn.Module, cluster_centers):
        super().__init__(model)
        self.cluster_centers = cluster_centers

    def to(self, device: torch.device) -> None:
        self.cluster_centers = self.cluster_centers.to(device)
        return super().to(device)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:

        self.model.eval()
        embeddings = self.model.forward(images / 255)
        distances = torch.cdist(embeddings, self.cluster_centers)

        prob = torch.reciprocal(distances + 0.01) ** 7
        prob /= prob.sum(1, keepdim=True)

        probabilities, assignments = prob.max(1)

        return assignments + 1, probabilities

    @classmethod
    def load(cls, path: Path):
        assert path.is_dir()
        cluster_centers = torch.from_numpy(
            np.loadtxt(path / cls.cluster_centers_filename, delimiter=",")
        )
        model = ResNet18.from_file(path / cls.model_weights_filename)
        return cls(model, cluster_centers)

    def save(self, path: Path, **extra_data):
        assert path.is_dir()
        np.savetxt(
            path / self.cluster_centers_filename,
            self.cluster_centers.numpy(force=True),
            fmt="%11.5f",
            delimiter=",",
        )
        return super().save(path, **extra_data)


def load_identifier_model(
    path: Path, image_size: Sequence[int] | None
) -> IdentifierCNN | IdentifierContrastive:
    model = None
    try:
        model = IdentifierContrastive.load(path)
    except FileNotFoundError:
        logging.info("Contrastive model not found in %s", path)

    assert image_size is not None
    model = IdentifierCNN.load(image_size, path)

    assert model is not None
    return model
