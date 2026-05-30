"""
SAM 3 (Segment Anything Model 3) integration for idtrackerai.

This subpackage provides text-prompted video segmentation using Meta's SAM 3
via the **ultralytics** library, replacing the legacy threshold-based
segmentation pipeline with learned, concept-driven mask generation.

SAM 3 performs unified detection + segmentation + tracking from a single
text prompt — no separate object detector is required.

Usage:
    Set segmentation_method="sam3" and sam3_text_prompt="zebrafish"
    in your parameter file or via CLI arguments.
"""

from .predictor import SAM3AnimalSegmenter
from .frame_extractor import extract_frames_to_directory, cleanup_frames_directory
from .mask_to_blobs import masks_to_blobs

__all__ = [
    "SAM3AnimalSegmenter",
    "extract_frames_to_directory",
    "cleanup_frames_directory",
    "masks_to_blobs",
]
