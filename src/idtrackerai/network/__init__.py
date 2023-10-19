"""isort:skip_file"""

# NetworkParams should be loaded before LearnerClassification
from torch.backends import cudnn

from .utils import fc_weights_reinit, weights_xavier_init, DEVICE
from .network_params import NetworkParams
from .learners import LearnerClassification
from .train import train, evaluate

cudnn.benchmark = True  # make it train faster

__all__ = [
    "evaluate",
    "LearnerClassification",
    "train",
    "weights_xavier_init",
    "fc_weights_reinit",
    "NetworkParams",
    "DEVICE",
]
