"""Detectron2 instance segmentation backend for idtrackerai.

This submodule provides a Detectron2-based animal segmentation pipeline
as an alternative to the legacy threshold and SAM 3 backends.

Usage::

    from idtrackerai.base.animals_detection.detectron2 import (
        Detectron2AnimalSegmenter,
    )
"""

from .predictor import Detectron2AnimalSegmenter

__all__ = ["Detectron2AnimalSegmenter"]
