import json
import logging
from abc import ABC
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18(ResNet):

    __call__: Callable[[Tensor], Tensor]

    def __init__(self, n_channels_in: int = 1, n_dimensions_out: int = 8) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=n_dimensions_out)
        if n_channels_in != 3:
            # adapt first conv layer to our single channel images (not RGB)
            self.conv1 = torch.nn.Conv2d(
                n_channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    @classmethod
    def from_file(cls, path: Path | str):
        logging.info(f"Loading {cls.__name__} weights from {path}")
        model_state_dict = torch.load(path, weights_only=True)
        n_dimensions_out = len(model_state_dict["fc.weight"])
        n_channels_in = model_state_dict["conv1.weight"].shape[1]
        model = cls(n_channels_in, n_dimensions_out)
        model.load_state_dict(model_state_dict)
        return model

    def compile(self) -> None:
        logging.info("Compiling the model %s", self.__class__.__name__)
        return super().compile()


class IdCNN(nn.Module):
    __call__: Callable[[Tensor], Tensor]
    legacy_normalization: bool = False
    """Before v6.0.0, the models were trained with a different
    normalization scheme. This flag can reenable it for backward compatibility."""

    def __init__(self, input_shape: Sequence[int], out_dim: int):
        logging.info(f"Creating {self.__class__.__name__} model")
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
        if self.legacy_normalization:
            x -= x.mean((1, 2, 3), keepdim=True)
            x /= x.std((1, 2, 3), keepdim=True)
        return self.layers(x)

    def reinitilaize(self):
        logging.info(f"Reinitializing {self.__class__.__name__}")

        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(init_func)

    def fully_connected_reinitialization(self):
        logging.info(
            f"Reinitializing only {self.__class__.__name__} fully connected layers"
        )

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

        model_params_path = model_path.parent / "model_params.json"
        if model_params_path.is_file():
            model_params = json.loads(model_params_path.read_text())
            if "version" in model_params:
                version = tuple(
                    map(int, model_params["version"].split("a")[0].split("."))
                )
            else:
                # before 6.0.0 the version was not included in the model_params.json
                version = None
        else:
            if "_" in model_path.parent.name:
                # prior to 6.0.0, the accumulation folders had an index like "accumulation_0"
                version = None
            else:
                version = (6, 0, 0)  # at least 6.0.0

        logging.info("Load model weights from %s", model_path)
        # The path to model file (*.best_model.pth). Do NOT use checkpoint file here
        # model_path can contain metadata in previous versions of idtrackerai, that's why weights_only=False
        model_state: dict = torch.load(model_path, weights_only=False)
        model_state.pop("val_acc", None)
        model_state.pop("test_acc", None)
        model_state.pop("ratio_accumulated", None)

        if "fc2.weight" in model_state:
            n_classes = len(model_state["fc2.weight"])
        else:
            n_classes = len(model_state["layers.11.weight"])

        model = cls(image_size, n_classes)

        if version is None or version < (6, 0, 0):
            # The model was trained with legacy normalization, we need to set it to True
            model.legacy_normalization = True

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

    def compile(self) -> None:
        logging.info("Compiling the model %s", self.__class__.__name__)
        return super().compile()


class IdentifierBase(ABC):
    model: nn.Module
    model_weights_filename: str

    def compile(self) -> None:
        return self.model.compile()

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def forward(self, images: Tensor) -> Tensor:
        "Takes a tensor images of size (Batch size, 1 ,Height, Width) in the range [0,1] and outputs another tensor of size (Batch size, n_animals) for the predicted probabilities"
        raise NotImplementedError

    def __call__(self, images: Tensor) -> Tensor:
        return self.forward(images)

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError

    def save(self, path: Path) -> None:
        assert path.is_dir()
        path = path / self.model_weights_filename
        logging.info("Saving %s at %s", self.__class__.__name__, path)
        torch.save(self.model.state_dict(), path)


class IdentifierIdCNN(IdentifierBase):
    model: IdCNN  # pyright: ignore[reportIncompatibleVariableOverride] IdCNN is subclass of torch.nn.Module
    model_weights_filename: str = "identifier_cnn.model.pt"

    def forward(self, images: Tensor) -> Tensor:
        return nn.functional.softmax(self.model(images), dim=1)

    def save(self, path: Path) -> None:
        assert path.is_dir()
        return super().save(path)

    @classmethod
    def load(cls, image_size: Sequence[int], model_path: Path):
        return cls(IdCNN.load(image_size, model_path))


class IdentifierContrastive(IdentifierBase):
    cluster_centers: Tensor
    model_weights_filename: str = "identifier_contrastive.model.pt"
    cluster_centers_filename: str = "identifier_contrastive.cluster_centers.csv"

    def __init__(self, model: nn.Module, cluster_centers):
        super().__init__(model)
        self.cluster_centers = cluster_centers

    def to(self, device: torch.device):
        self.cluster_centers = self.cluster_centers.to(device)
        return super().to(device)

    def forward(self, images: Tensor) -> Tensor:
        embeddings = self.model(images)
        distances = torch.cdist(embeddings, self.cluster_centers)

        prob = torch.reciprocal(distances + 0.01) ** 7
        prob /= prob.sum(1, keepdim=True)

        return prob

    @classmethod
    def load(cls, path: Path | str):
        path = Path(path)
        assert path.is_dir()
        cluster_centers = torch.from_numpy(
            np.loadtxt(
                path / cls.cluster_centers_filename, delimiter=",", dtype=np.float32
            )
        )
        model = ResNet18.from_file(path / cls.model_weights_filename)
        return cls(model, cluster_centers)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        assert path.is_dir()
        np.savetxt(
            path / self.cluster_centers_filename,
            self.cluster_centers.numpy(force=True),
            fmt="%11.5f",
            delimiter=",",
        )
        return super().save(path)


def load_identifier_model(
    path: Path, image_size: Sequence[int] | None
) -> IdentifierIdCNN | IdentifierContrastive:
    try:
        model = IdentifierContrastive.load(path)
    except FileNotFoundError:
        logging.info("Contrastive model not found in %s", path)
    else:
        return model

    assert image_size is not None, "Image size must be provided for IdCNN model"
    model = IdentifierIdCNN.load(image_size, path)

    assert model is not None
    return model
