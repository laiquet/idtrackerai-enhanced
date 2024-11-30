"""isort:skip_file"""

from torch.backends import cudnn

from .device import DEVICE
from .models import (
    CNN,
    IdentifierBase,
    IdentifierCNN,
    IdentifierContrastive,
    ResNet18,
    load_identifier_model,
)
from .train import (
    evaluate,
    evaluate_only_acc,
    StopTraining,
    train_loop,
    ImageDataset,
    get_dataloader,
    get_predictions,
    get_onthefly_dataloader,
    DataLoaderWithLabels,
)

cudnn.benchmark = True  # make it train faster

__all__ = [
    "DataLoaderWithLabels",
    "evaluate",
    "load_identifier_model",
    "IdentifierCNN",
    "DEVICE",
    "CNN",
    "ResNet18",
    "evaluate_only_acc",
    "StopTraining",
    "train_loop",
    "ImageDataset",
    "get_dataloader",
    "IdentifierContrastive",
    "get_predictions",
    "get_onthefly_dataloader",
    "IdentifierBase",
]
