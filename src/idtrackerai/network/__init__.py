"""isort:skip_file"""

# NetworkParams should be loaded before LearnerClassification
from torch.backends import cudnn

from .utils import (
    fully_connected_reinitialization,
    full_reinitialization,
    DEVICE,
    DataLoaderWithLabels,
)
from .network_params import NetworkParams
from .models import CNN
from .learners import LearnerClassification
from .train import train, evaluate, evaluate_only_acc

cudnn.benchmark = True  # make it train faster

__all__ = [
    "evaluate",
    "LearnerClassification",
    "train",
    "full_reinitialization",
    "fully_connected_reinitialization",
    "NetworkParams",
    "DEVICE",
    "CNN",
    "evaluate_only_acc",
    "DataLoaderWithLabels",
]
