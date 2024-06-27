"""isort:skip_file"""

# NetworkParams should be loaded before LearnerClassification
from torch.backends import cudnn

from .utils import DEVICE, DataLoaderWithLabels
from .network_params import NetworkParams
from .models import CNN, IdentifierBase, IdentifierCNN, IdentifierContrastive, ResNet18
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
    "IdentifierCNN",
    "NetworkParams",
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
