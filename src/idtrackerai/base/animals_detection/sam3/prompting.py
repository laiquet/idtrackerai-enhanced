"""
Hybrid prompting — fallback to legacy threshold-based segmentation.

When SAM 3's text prompt misses some animals (e.g. detects 6 out of 8),
this module runs the legacy threshold-based segmentation on specific
frames and generates point prompts for the missing animals.
"""

import logging

import cv2
import numpy as np

from idtrackerai.base.animals_detection.segmentation import process_frame

logger = logging.getLogger("idtrackerai.sam3.prompting")


def generate_hybrid_prompts(
    frame: np.ndarray,
    intensity_ths: tuple | list,
    area_ths: tuple | list,
    number_of_animals: int,
    sam3_detections: dict[int, np.ndarray],
    ROI_mask: np.ndarray | None = None,
    bkg_model: np.ndarray | None = None,
    frame_idx: int = 0,
) -> dict[int, dict] | None:
    """Generate point prompts for animals missed by SAM 3.

    If SAM 3's text-prompted detection found fewer animals than
    expected, this function runs the legacy threshold segmentation
    pipeline on the given frame and identifies blob centroids that
    don't overlap with any SAM 3 mask. Those centroids are returned
    as point prompts that can be fed back into SAM 3.

    Parameters
    ----------
    frame : np.ndarray
        Grayscale frame to run threshold segmentation on.
    intensity_ths : tuple or list
        Intensity thresholds (min, max).
    area_ths : tuple or list
        Area thresholds (min, max).
    number_of_animals : int
        Expected number of animals.
    sam3_detections : dict
        Dict mapping ``obj_id -> binary_mask`` from SAM 3's
        initial detection pass.
    ROI_mask : np.ndarray, optional
        Region of interest mask.
    bkg_model : np.ndarray, optional
        Background model for subtraction.
    frame_idx : int
        Frame index for the prompt (default 0 for first frame).

    Returns
    -------
    additional_prompts : dict or None
        Dict mapping new ``obj_id -> {"frame_idx": int, "points":
        np.array, "labels": np.array}`` for each missed animal.
        Returns ``None`` if SAM 3 found all animals.
    """
    num_sam3_detections = len(sam3_detections)

    if num_sam3_detections >= number_of_animals:
        logger.info(
            f"SAM 3 detected {num_sam3_detections}/{number_of_animals} "
            f"animals — no hybrid prompts needed"
        )
        return None

    missing_count = number_of_animals - num_sam3_detections
    logger.info(
        f"SAM 3 detected {num_sam3_detections}/{number_of_animals} animals. "
        f"Generating {missing_count} hybrid point prompts from "
        f"threshold segmentation."
    )

    # Run legacy threshold segmentation on this frame
    _areas, contours, _gray = process_frame(
        frame.copy(),
        intensity_ths=intensity_ths,
        area_ths=area_ths,
        ROI_mask=ROI_mask,
        bkg_model=bkg_model,
    )

    if not contours:
        logger.warning(
            "Threshold segmentation found no blobs on this frame. "
            "Cannot generate hybrid prompts."
        )
        return None

    # Compute centroids from contours
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy))

    if not centroids:
        return None

    # Build a combined SAM 3 mask to identify coverage
    h, w = frame.shape[:2]
    sam3_combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in sam3_detections.values():
        sam3_combined_mask = np.bitwise_or(
            sam3_combined_mask, mask.astype(np.uint8)
        )

    # Find threshold centroids that DON'T overlap with SAM 3 masks
    unmatched_centroids = []
    for centroid in centroids:
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        # Check if this centroid falls inside any SAM 3 mask
        if (
            0 <= cy < h
            and 0 <= cx < w
            and sam3_combined_mask[cy, cx] == 0
        ):
            unmatched_centroids.append(centroid)

    if not unmatched_centroids:
        logger.info(
            "All threshold-detected centroids overlap with SAM 3 masks. "
            "No additional prompts to generate."
        )
        return None

    # Generate point prompts for the unmatched centroids
    # Assign new obj_ids starting after the highest SAM 3 obj_id
    max_sam3_id = max(sam3_detections.keys()) if sam3_detections else -1
    additional_prompts: dict[int, dict] = {}

    for i, centroid in enumerate(unmatched_centroids[:missing_count]):
        new_obj_id = max_sam3_id + 1 + i
        additional_prompts[new_obj_id] = {
            "frame_idx": frame_idx,
            "points": np.array(
                [[centroid[0], centroid[1]]], dtype=np.float32
            ),
            "labels": np.array([1], dtype=np.int32),  # 1 = foreground
        }

    logger.info(
        f"Generated {len(additional_prompts)} hybrid prompts for "
        f"missing animals"
    )

    return additional_prompts
