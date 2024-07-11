"""isort:skip_file"""

from torch.backends import cudnn

from .utils import DEVICE, DataLoaderWithLabels
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
)

cudnn.benchmark = True  # make it train faster

__all__ = [
    "evaluate",
    "load_identifier_model",
    "IdentifierCNN",
    "DEVICE",
    "CNN",
    "ResNet18",
    "evaluate_only_acc",
    "DataLoaderWithLabels",
    "StopTraining",
    "train_loop",
    "ImageDataset",
    "get_dataloader",
    "IdentifierContrastive",
    "get_predictions",
    "get_onthefly_dataloader",
    "IdentifierBase",
]
