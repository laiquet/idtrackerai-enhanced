"""
Frame extraction utilities for SAM 3 video segmentation.

SAM 3's video predictor requires frames as JPEG files in a directory.
This module extracts frames from videos and writes them as numbered
JPEG files.
"""

import gc
import logging
import os
import shutil

import cv2
import numpy as np

logger = logging.getLogger("idtrackerai.sam3.frame_extractor")


def extract_frames_to_directory(
    video_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    start_frame: int = 0,
    end_frame: int | None = None,
    roi_mask: np.ndarray | None = None,
) -> tuple[str, int]:
    """Extract video frames as numbered JPEGs for SAM 3.

    SAM 3's video predictor accepts a directory of sequential JPEG
    frames. This function reads frames from the video source and
    writes them as numbered JPEG files (00000.jpg, 00001.jpg, ...).

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.
    output_dir : str or Path
        Directory where JPEG frames will be written. Created if it
        does not exist.
    start_frame : int
        First frame to extract (inclusive, 0-indexed in the video).
    end_frame : int, optional
        Last frame to extract (exclusive). If None, extracts to the
        end of the video.
    roi_mask : np.ndarray, optional
        Binary mask (same shape as frame). Pixels outside the mask
        (where mask == 0) are zeroed out at extraction time.

    Returns
    -------
    frames_dir : str
        Absolute path to the directory containing the extracted frames.
    num_frames : int
        Total number of frames successfully extracted.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_frames = end_frame - start_frame

    logger.info(
        f"Extracting {total_frames} frames from "
        f"[{start_frame}, {end_frame}) to {output_dir}"
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    num_extracted = 0
    apply_enhancement = False
    checked_contrast = False

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning(
                f"Could not read frame at index {start_frame + i}. "
                f"Stopping extraction at {num_extracted} frames."
            )
            break

        # Apply ROI mask — zero out pixels outside the region of interest
        if roi_mask is not None:
            frame = cv2.bitwise_and(
                frame,
                frame,
                mask=roi_mask.astype(np.uint8),
            )

        # Dynamic contrast check on first frame
        if not checked_contrast:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast_score = gray_frame.std()
            logger.info(f"SAM 3: Initial frame contrast score is {contrast_score:.2f}")
            if contrast_score < 25.0:  # Threshold for low contrast
                apply_enhancement = True
                logger.info("SAM 3: Low contrast detected. Enabling CLAHE + Bilateral contrast enhancement.")
            checked_contrast = True

        if apply_enhancement:
            # Apply CLAHE on the L (Luminance) channel of LAB space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            
            # Apply Bilateral Filter to suppress background noise while keeping borders sharp
            frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # Write as numbered JPEG
        frame_filename = os.path.join(output_dir, f"{i:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        num_extracted += 1

    cap.release()
    gc.collect()

    logger.info(
        f"Extracted {num_extracted} frames to {output_dir}"
    )

    return output_dir, num_extracted


def cleanup_frames_directory(frames_dir: str | os.PathLike) -> None:
    """Remove the directory of extracted JPEG frames.

    Parameters
    ----------
    frames_dir : str or Path
        Path to the directory to remove.
    """
    frames_dir = str(frames_dir)
    if os.path.isdir(frames_dir):
        logger.info(f"Cleaning up frames directory: {frames_dir}")
        shutil.rmtree(frames_dir)
