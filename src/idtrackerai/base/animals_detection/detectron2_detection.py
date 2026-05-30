"""Detectron2 animals detection API.

This module provides ``detectron2_detection_API``, a drop-in replacement
for ``animals_detection_API`` that uses Detectron2 instance segmentation
instead of the legacy threshold-based pipeline.

The output is a standard ``ListOfBlobs`` object — identical in structure
to what ``animals_detection_API`` and ``sam3_detection_API`` produce — so
all downstream processing (crossing detection, fragmentation, identity
assignment, post-processing) works without modification.
"""

import gc
import logging
from io import BytesIO
from pathlib import Path

import cv2
import h5py
import numpy as np

from idtrackerai import Blob, ListOfBlobs, Session

from .detectron2 import Detectron2AnimalSegmenter
from .sam3.mask_to_blobs import masks_to_blobs
from .segmentation import get_bbox_image, to_gray_scale

logger = logging.getLogger("idtrackerai.detectron2_detection")


def detectron2_detection_API(session: Session) -> ListOfBlobs:
    """Segment the video using Detectron2 and return a ListOfBlobs.

    This function mirrors ``animals_detection_API()`` and
    ``sam3_detection_API()`` but uses Detectron2 instance segmentation.

    Pipeline:
        1. Initialize Detectron2 predictor with config + weights
        2. Run per-frame inference → binary masks per detected instance
        3. Apply IoU temporal smoothing for consistent tracking IDs
        4. Convert masks → Blob objects per frame (with bbox images)
        5. Package as ListOfBlobs

    Parameters
    ----------
    session : Session
        The session object containing video paths, Detectron2 parameters,
        and output paths.

    Returns
    -------
    list_of_blobs : ListOfBlobs
    """
    logging.info("Starting Detectron2 video segmentation")

    # Validate mandatory Detectron2 parameters
    if not session.detectron2_config:
        raise ValueError(
            "Detectron2 config file is required. "
            "Set 'detectron2_config' in session parameters."
        )
    if not session.detectron2_weights:
        raise ValueError(
            "Detectron2 weights file is required. "
            "Set 'detectron2_weights' in session parameters."
        )
    if not session.detectron2_class_names:
        raise ValueError(
            "Detectron2 class name(s) required. "
            "Set 'detectron2_class_names' in session parameters "
            "(e.g. ['fish'] or ['animal'])."
        )

    # Setup bbox images storage — same pattern as legacy/SAM3 pipeline
    if session.bounding_box_images_in_ram:
        from idtrackerai.utils import remove_dir

        remove_dir(session.bbox_images_folder)
    else:
        from idtrackerai.utils import create_dir

        create_dir(session.bbox_images_folder, remove_existing=True)

    # Step 1: Run Detectron2 segmentation
    start_frame = session.episodes[0].global_start
    end_frame = session.episodes[-1].global_end
    video_path = str(session.video_paths[0])
    num_frames = end_frame - start_frame

    segmenter = Detectron2AnimalSegmenter(
        config_path=session.detectron2_config,
        weights_path=session.detectron2_weights,
        confidence_threshold=session.detectron2_confidence_threshold,
        device="cuda",
    )

    video_segments = segmenter.segment_video(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        roi_mask=session.ROI_mask,
        class_names=session.detectron2_class_names,
    )

    segmenter.release()

    # Step 2: IoU temporal smoothing for consistent tracking IDs
    # Reuse the same function from sam3_detection
    from .sam3_detection import _smooth_tracking_ids_by_iou

    video_segments = _smooth_tracking_ids_by_iou(video_segments, num_frames)

    # Step 3: Validate detections
    if session.n_animals > 0:
        frames_too_few = sum(
            1
            for f in range(num_frames)
            if len(video_segments.get(f, {})) < session.n_animals
        )
        frames_too_many = sum(
            1
            for f in range(num_frames)
            if len(video_segments.get(f, {})) > session.n_animals
        )
        if frames_too_few > 0:
            pct = frames_too_few / num_frames * 100
            logging.warning(
                f"Detectron2: {frames_too_few} frames ({pct:.1f}%) have "
                f"fewer than {session.n_animals} detections"
            )
        if frames_too_many > 0:
            pct = frames_too_many / num_frames * 100
            logging.warning(
                f"Detectron2: {frames_too_many} frames ({pct:.1f}%) have "
                f"more than {session.n_animals} detections"
            )

    # Step 4: Convert masks to Blob objects
    logging.info("Converting Detectron2 masks to Blob objects")
    blobs_in_video = _masks_to_blobs_all_frames(
        session, video_segments, num_frames, start_frame, video_path
    )

    gc.collect()

    # Pad to full video length
    frames_before = start_frame
    frames_after = session.number_of_frames - end_frame
    blobs_in_video = (
        [[] for _ in range(frames_before)]
        + blobs_in_video
        + [[] for _ in range(frames_after)]
    )

    list_of_blobs = ListOfBlobs(blobs_in_video)
    assert len(list_of_blobs) == session.number_of_frames

    n_detected_blobs = list_of_blobs.number_of_blobs
    trackable_frames = (
        sum(end - start for start, end in session.tracking_intervals)
        if session.tracking_intervals
        else session.number_of_frames
    )
    logging.info(
        f"Detectron2: {n_detected_blobs} detected blobs in total, "
        f"an average of {n_detected_blobs / trackable_frames:.1f} blobs per frame"
    )

    # Run segmentation quality check
    from .animals_detection import check_segmentation

    check_segmentation(session, list_of_blobs)

    return list_of_blobs


def _masks_to_blobs_all_frames(
    session: Session,
    video_segments: dict[int, dict[int, np.ndarray]],
    num_frames: int,
    start_frame: int,
    video_path: str,
) -> list[list[Blob]]:
    """Convert all per-frame Detectron2 masks to Blob objects.

    Reuses the same ``masks_to_blobs()`` bridge as the SAM 3 pipeline,
    with identical HDF5 bounding box image storage per episode.
    """
    from idtrackerai.utils import track

    # Set up per-episode bbox image storage
    if not session.bounding_box_images_in_ram:
        for episode in session.episodes:
            episode.bbox_images = (
                session.bbox_images_folder / f"bbox_images_{episode.index}.h5"
            )

    cap = cv2.VideoCapture(video_path)
    blobs_in_video: list[list[Blob]] = []

    # Build frame → episode mapping
    frame_to_episode: dict[int, int] = {}
    for ep_idx, episode in enumerate(session.episodes):
        for f in range(episode.global_start, episode.global_end):
            frame_to_episode[f] = ep_idx

    # Open one H5 file per episode
    episode_h5_files: dict[int, h5py.File] = {}
    try:
        for ep_idx, episode in enumerate(session.episodes):
            if session.bounding_box_images_in_ram:
                buf = BytesIO()
                episode.bbox_images = buf
                episode_h5_files[ep_idx] = h5py.File(buf, "w")
            else:
                episode_h5_files[ep_idx] = h5py.File(episode.bbox_images, "w")

        for frame_idx in track(range(num_frames), "Detectron2 → Blobs"):
            global_frame_number = start_frame + frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame_number)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Could not read frame {global_frame_number}")
                blobs_in_video.append([])
                continue

            gray = to_gray_scale(frame)
            masks = video_segments.get(frame_idx, {})

            if masks:
                ep_idx = frame_to_episode.get(global_frame_number)
                if ep_idx is not None and ep_idx in episode_h5_files:
                    h5_file = episode_h5_files[ep_idx]
                else:
                    h5_file = episode_h5_files[0]

                blobs = masks_to_blobs(
                    frame=gray,
                    masks=masks,
                    frame_number=global_frame_number,
                    bbox_images_file=h5_file,
                )
                blobs_in_video.append(blobs)
            else:
                blobs_in_video.append([])

    finally:
        for h5_file in episode_h5_files.values():
            h5_file.close()

    cap.release()

    return blobs_in_video
