"""isort:skip_file"""

# NetworkParams should be loaded before LearnerClassification
from torch.backends import cudnn

from .utils import DEVICE, DataLoaderWithLabels
from .network_params import NetworkParams
from .models import CNN
from .learners import LearnerClassification
from .train import (
    train,
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
    "LearnerClassification",
    "train",
    "NetworkParams",
    "DEVICE",
    "CNN",
    "evaluate_only_acc",
    "DataLoaderWithLabels",
    "StopTraining",
    "train_loop",
    "ImageDataset",
    "get_dataloader",
    "get_predictions",
    "get_onthefly_dataloader",
]
