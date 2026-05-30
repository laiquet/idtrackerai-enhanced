"""
Segmentation mask → idtrackerai Blob conversion bridge.

This module converts binary segmentation masks into idtrackerai
``Blob`` objects, reusing the existing blob creation infrastructure
from ``segmentation.py``.

This is the **critical integration point** — it ensures that
masks from any backend (SAM 3, Detectron2, etc.) produce Blob
objects indistinguishable from those created by the legacy
threshold-based pipeline, so all downstream processing
(crossing detection, fragmentation, identity assignment, etc.)
works without modification.
"""

import logging

import cv2
import h5py
import numpy as np

from idtrackerai import Blob
from idtrackerai.base.animals_detection.segmentation import get_bbox_image

logger = logging.getLogger("idtrackerai.mask_to_blobs")


def masks_to_blobs(
    frame: np.ndarray,
    masks: dict[int, np.ndarray],
    frame_number: int,
    bbox_images_file: h5py.File,
) -> list[Blob]:
    """Convert binary segmentation masks to idtrackerai Blob objects.

    This function takes per-object binary masks from any segmentation
    backend (SAM 3, Detectron2, etc.) and converts them into standard
    ``Blob`` objects that the rest of the idtrackerai pipeline can
    process.

    Pipeline:
        1. For each binary mask, apply morphological cleanup (close + open)
        2. Find contours using ``cv2.findContours``
        3. Keep only the largest contour per mask (filters minor artifacts)
        4. Simplify the contour with Douglas-Peucker approximation
        5. Extract bounding box image using existing ``get_bbox_image()``
        6. Create ``Blob(contour, frame_number, bbox_img_id)``

    Parameters
    ----------
    frame : np.ndarray
        Grayscale frame from which the blob image crops are extracted.
    masks : dict
        Dict mapping ``obj_id`` (int) to binary mask (np.ndarray of
        shape ``(H, W)`` with values 0 or 1).
    frame_number : int
        Global frame number in the full video.
    bbox_images_file : h5py.File
        Open HDF5 file handle for writing bounding box images.

    Returns
    -------
    blobs_in_frame : list of Blob
        List of ``Blob`` objects, one per detected animal mask.
    """
    if not masks:
        return []

    blobs_in_frame: list[Blob] = []
    blob_index = 0

    for obj_id in sorted(masks.keys()):
        mask = masks[obj_id]
        # Ensure mask is uint8 with values 0/255 for findContours
        mask_255 = (mask * 255).astype(np.uint8)

        # Morphological cleanup: fill small holes (close) then
        # remove fringe noise (open) to stabilize mask boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
        mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, kernel)

        found_contours = cv2.findContours(
            mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # OpenCV 4.x returns (contours, hierarchy)
        cnt_list = found_contours[0]

        if not cnt_list:
            logger.warning(
                f"Frame {frame_number}, obj_id {obj_id}: "
                f"no contours found in mask, skipping"
            )
            continue

        # Keep only the largest contour per object
        # (filters out minor mask artifacts/noise)
        largest_contour = max(cnt_list, key=cv2.contourArea)

        # Simplify contour with Douglas-Peucker approximation to
        # reduce single-pixel boundary jitter between frames
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Skip if the contour is too small (< 3 points)
        if len(largest_contour) < 3:
            logger.warning(
                f"Frame {frame_number}, obj_id {obj_id}: "
                f"contour has < 3 points, skipping"
            )
            continue

        # Squeeze the contour to shape [n_points, 2] — same as upstream
        contour = np.squeeze(largest_contour)

        # Store bounding box image — same pattern as legacy segmentation
        dataset_name = f"{frame_number}-{blob_index}"
        bbox_images_file.create_dataset(
            dataset_name, data=get_bbox_image(frame, contour)
        )

        blobs_in_frame.append(Blob(contour, frame_number, dataset_name))
        blobs_in_frame[-1].segment_track_id = obj_id
        blob_index += 1

    logger.debug(
        f"Frame {frame_number}: converted {len(blobs_in_frame)} masks "
        f"to Blob objects"
    )

    return blobs_in_frame
