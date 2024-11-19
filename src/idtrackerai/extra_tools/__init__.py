"""idtracker.ai tools, which have a dedicated command line main entrypoint, can also be called from here as Python functions."""

import logging

try:
    # allow loading extra_tools even if PyTorch is not installed,
    # making the import of idmatcher.ai to fail
    from .idmatcherai import idmatcherai
except ModuleNotFoundError as exc:
    logging.error(f"Could not automatically import idmatcher.ai. {exc}")
    pass
from .cluster_inspection import inspect_clusters
from .validator import idtrackerai_validate
from .video_generator import generate_individual_video, generate_trajectories_video

__all__ = [
    "idmatcherai",
    "idtrackerai_validate",
    "generate_individual_video",
    "generate_trajectories_video",
    "inspect_clusters",
]
