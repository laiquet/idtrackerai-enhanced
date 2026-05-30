"""
SAM 3 animals detection API (ultralytics backend).

This module provides ``sam3_detection_API``, a drop-in replacement for
``animals_detection_API`` that uses Meta's SAM 3 — via the **ultralytics**
library — for text-prompted video segmentation instead of the legacy
threshold-based pipeline.

SAM 3 performs unified detection + segmentation + tracking from a single
text prompt, so no separate object detector (GroundingDINO) is needed.

The output is a standard ``ListOfBlobs`` object — identical in structure
to what ``animals_detection_API`` produces — so all downstream processing
(crossing detection, fragmentation, identity assignment, post-processing)
works without modification.
"""

import gc
import logging
import os
from io import BytesIO
from pathlib import Path

import cv2
import h5py
import numpy as np

from idtrackerai import Blob, ListOfBlobs, Session

from .sam3 import SAM3AnimalSegmenter, cleanup_frames_directory, extract_frames_to_directory
from .sam3.mask_to_blobs import masks_to_blobs
from .segmentation import get_bbox_image, to_gray_scale

logger = logging.getLogger("idtrackerai.sam3_detection")


def sam3_detection_API(session: Session) -> ListOfBlobs:
    """Segment the video using SAM 3 (ultralytics) and return a ListOfBlobs.

    This function mirrors ``animals_detection_API()`` but uses SAM 3
    text-prompted segmentation instead of intensity/area thresholding.

    Pipeline:
        1. Extract video frames as numbered JPEGs
        2. Run ultralytics SAM 3 with text prompt → per-frame binary masks
        3. Convert masks → Blob objects per frame (with bbox images)
        4. Package as ListOfBlobs

    Parameters
    ----------
    session : Session
        The session object containing video paths, SAM 3 parameters,
        and output paths.

    Returns
    -------
    list_of_blobs : ListOfBlobs
    """
    logging.info("Starting SAM 3 video segmentation (ultralytics backend)")

    # Setup bbox images storage — same pattern as legacy pipeline
    if session.bounding_box_images_in_ram:
        from idtrackerai.utils import remove_dir

        remove_dir(session.bbox_images_folder)
    else:
        from idtrackerai.utils import create_dir

        create_dir(session.bbox_images_folder, remove_existing=True)

    # Step 1: Extract frames to JPEG directory
    frames_dir = str(session.session_folder / "sam3_frames")
    start_frame = session.episodes[0].global_start
    end_frame = session.episodes[-1].global_end
    video_path = str(session.video_paths[0])

    frames_dir, num_frames = extract_frames_to_directory(
        video_path=video_path,
        output_dir=frames_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        roi_mask=session.ROI_mask,
    )

    # Step 2: Run SAM 3 with text prompt via ultralytics
    # Auto-discover sam3.pt from weights/ directory or let ultralytics resolve
    weights_dir = Path(__file__).resolve().parents[4] / "weights"
    sam3_path = weights_dir / "sam3.pt"
    checkpoint = str(sam3_path) if sam3_path.is_file() else "sam3.pt"
    logging.info(f"Using SAM 3 checkpoint: {checkpoint}")

    segmenter = SAM3AnimalSegmenter(checkpoint=checkpoint, device="cuda")
    video_segments = segmenter.segment_video(
        frames_dir=frames_dir,
        text_prompt=session.sam3_text_prompt,
        number_of_animals=session.n_animals,
        confidence_threshold=session.sam3_confidence_threshold,
        video_path=video_path,
    )

    # Step 2b: Hybrid prompting — recover animals missed by text-only SAM 3
    if session.n_animals > 0:
        video_segments = _hybrid_prompting_refinement(
            session=session,
            segmenter=segmenter,
            video_segments=video_segments,
            frames_dir=frames_dir,
            video_path=video_path,
            start_frame=start_frame,
            num_frames=num_frames,
        )

    # Step 2c: IoU-based temporal ID smoothing
    #   Ensures consistent tracking IDs across frames by matching masks
    #   between consecutive frames using Intersection-over-Union (IoU).
    video_segments = _smooth_tracking_ids_by_iou(video_segments, num_frames)

    # Release SAM 3 GPU memory before blob conversion
    segmenter.release()

    # Step 3: Convert SAM 3 masks to Blob objects
    logging.info("Converting SAM 3 masks to Blob objects")
    blobs_in_video = _masks_to_blobs_all_frames(
        session, video_segments, num_frames, start_frame, video_path
    )

    # Step 4: Cleanup extracted frames
    cleanup_frames_directory(frames_dir)
    gc.collect()

    # Pad to full video length (empty lists for frames outside tracking range)
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
        f"SAM 3: {n_detected_blobs} detected blobs in total, "
        f"an average of {n_detected_blobs / trackable_frames:.1f} blobs per frame"
    )

    # Run the same segmentation check as legacy pipeline
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
    """Convert all per-frame masks to Blob objects.

    Opens the video once and iterates through all frames, converting
    each frame's SAM 3 masks into Blob objects with proper HDF5
    bounding box image storage (one file per episode, matching the
    legacy pipeline pattern).
    """
    from idtrackerai.utils import track

    # Set up per-episode bbox image storage — same pattern as legacy pipeline
    if not session.bounding_box_images_in_ram:
        for episode in session.episodes:
            episode.bbox_images = (
                session.bbox_images_folder / f"bbox_images_{episode.index}.h5"
            )

    cap = cv2.VideoCapture(video_path)
    blobs_in_video: list[list[Blob]] = []

    # Build a mapping: global_frame_number -> episode index for efficient lookup
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

        for frame_idx in track(range(num_frames), "SAM 3 → Blobs"):
            global_frame_number = start_frame + frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame_number)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Could not read frame {global_frame_number}")
                blobs_in_video.append([])
                continue

            # Convert to grayscale (same as legacy pipeline)
            gray = to_gray_scale(frame)

            masks = video_segments.get(frame_idx, {})

            if masks:
                # Find the H5 file for this frame's episode
                ep_idx = frame_to_episode.get(global_frame_number)
                if ep_idx is not None and ep_idx in episode_h5_files:
                    h5_file = episode_h5_files[ep_idx]
                else:
                    # Fallback: use first episode's file
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
        # Close all H5 files
        for h5_file in episode_h5_files.values():
            h5_file.close()

    cap.release()

    return blobs_in_video


def _hybrid_prompting_refinement(
    session: Session,
    segmenter: SAM3AnimalSegmenter,
    video_segments: dict[int, dict[int, np.ndarray]],
    frames_dir: str,
    video_path: str,
    start_frame: int,
    num_frames: int,
) -> dict[int, dict[int, np.ndarray]]:
    """Refine SAM 3 detections using hybrid prompting for missed animals.

    When SAM 3's text-prompt segmentation finds fewer animals than
    expected, this function uses the legacy threshold-based pipeline
    to locate the missing animals and feeds their centroids back into
    SAM 3 as foreground point prompts.

    Parameters
    ----------
    session : Session
        The session object containing threshold parameters.
    segmenter : SAM3AnimalSegmenter
        The SAM 3 segmenter instance (still loaded on GPU).
    video_segments : dict
        Initial SAM 3 detection results to refine in-place.
    frames_dir : str
        Directory containing extracted JPEG frames.
    video_path : str
        Path to the source video file.
    start_frame : int
        Global start frame offset.
    num_frames : int
        Total number of extracted frames.

    Returns
    -------
    video_segments : dict
        The (potentially refined) detection results.
    """
    from .sam3.prompting import generate_hybrid_prompts

    # Guard: hybrid prompting needs legacy threshold parameters
    if session.intensity_ths is None or session.area_ths is None:
        logger.info(
            "Hybrid prompting skipped: intensity_ths and/or area_ths are not "
            "configured. Set these legacy threshold parameters in your session "
            "config to enable automatic recovery of animals missed by SAM 3."
        )
        return video_segments

    # Identify frames where SAM 3 detected fewer animals than expected
    n_animals = session.n_animals
    deficient_frame_indices = [
        f_idx for f_idx in range(num_frames)
        if len(video_segments.get(f_idx, {})) < n_animals
    ]

    if not deficient_frame_indices:
        logger.info("Hybrid prompting: all frames have expected animal count, no refinement needed")
        return video_segments

    pct = len(deficient_frame_indices) / num_frames * 100
    logger.info(
        f"Hybrid prompting: {len(deficient_frame_indices)} frames ({pct:.1f}%) "
        f"have fewer than {n_animals} detections. Attempting recovery."
    )

    # Sample frames to run hybrid prompting on. We check the first
    # deficient frame plus every 100th deficient frame to generate
    # representative point prompts without excessive computation.
    sample_indices = [deficient_frame_indices[0]]
    for idx in deficient_frame_indices[1::100]:
        if idx not in sample_indices:
            sample_indices.append(idx)

    cap = cv2.VideoCapture(video_path)
    total_recovered = 0

    for frame_idx in sample_indices:
        global_frame = start_frame + frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        sam3_detections = video_segments.get(frame_idx, {})

        # Generate point prompts from legacy threshold segmentation
        hybrid_prompts = generate_hybrid_prompts(
            frame=frame,
            intensity_ths=session.intensity_ths,
            area_ths=session.area_ths,
            number_of_animals=n_animals,
            sam3_detections=sam3_detections,
            ROI_mask=session.ROI_mask,
            bkg_model=None,
            frame_idx=frame_idx,
        )

        if hybrid_prompts is None:
            continue

        # Collect all point prompts from hybrid results
        all_points = []
        all_labels = []
        for prompt_info in hybrid_prompts.values():
            all_points.append(prompt_info["points"])
            all_labels.append(prompt_info["labels"])

        if not all_points:
            continue

        points = np.concatenate(all_points, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Build the frame path for the extracted JPEG
        frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
        if not os.path.isfile(frame_path):
            logger.warning(f"Frame JPEG not found at {frame_path}, skipping")
            continue

        # Run SAM 3 with point prompts on this specific frame
        new_masks = segmenter.segment_frame_with_points(
            frame_path=frame_path,
            points=points,
            labels=labels,
            confidence_threshold=session.sam3_confidence_threshold,
        )

        if new_masks:
            # Merge new masks into existing detections for this frame
            existing = video_segments.get(frame_idx, {})
            max_existing_id = max(existing.keys()) if existing else 0
            for new_obj_idx, mask in new_masks.items():
                merged_id = max_existing_id + new_obj_idx
                existing[merged_id] = mask
                total_recovered += 1
            video_segments[frame_idx] = existing

    cap.release()

    if total_recovered > 0:
        logger.info(
            f"Hybrid prompting recovered {total_recovered} additional "
            f"animal masks across {len(sample_indices)} sampled frames"
        )
    else:
        logger.info("Hybrid prompting: no additional animals recovered")

    return video_segments


def _smooth_tracking_ids_by_iou(
    video_segments: dict[int, dict[int, np.ndarray]],
    num_frames: int,
    iou_threshold: float = 0.3,
) -> dict[int, dict[int, np.ndarray]]:
    """Reassign tracking IDs across frames using IoU-based matching.

    For each pair of consecutive frames, computes pairwise IoU between
    all masks and uses the Hungarian algorithm to find the optimal
    assignment.  Masks with IoU >= ``iou_threshold`` inherit the
    tracking ID from the previous frame; unmatched masks get new IDs.

    This step runs *after* all detection / prompting passes and *before*
    mask-to-blob conversion, ensuring that the ``obj_id`` keys in
    ``video_segments`` are temporally consistent before they become
    ``Blob.segment_track_id`` values.

    Parameters
    ----------
    video_segments : dict
        ``{frame_idx: {obj_id: binary_mask}}``
    num_frames : int
        Total number of frames.
    iou_threshold : float
        Minimum IoU to consider two masks as the same object.

    Returns
    -------
    smoothed : dict
        Same structure, with reassigned obj_id keys.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.warning(
            "scipy not available for IoU-based ID smoothing, skipping"
        )
        return video_segments

    smoothed: dict[int, dict[int, np.ndarray]] = {}
    next_id = 1

    # Process first frame — assign fresh sequential IDs
    first_masks = video_segments.get(0, {})
    if first_masks:
        new_first: dict[int, np.ndarray] = {}
        id_map: dict[int, int] = {}
        for old_id in sorted(first_masks.keys()):
            id_map[old_id] = next_id
            new_first[next_id] = first_masks[old_id]
            next_id += 1
        smoothed[0] = new_first
        prev_ids = list(new_first.keys())
        prev_masks = new_first
    else:
        smoothed[0] = {}
        prev_ids = []
        prev_masks = {}

    n_reassigned = 0

    for f_idx in range(1, num_frames):
        curr_masks_raw = video_segments.get(f_idx, {})
        if not curr_masks_raw or not prev_masks:
            smoothed[f_idx] = {}
            prev_ids = []
            prev_masks = {}
            continue

        curr_ids = sorted(curr_masks_raw.keys())
        curr_masks_list = [curr_masks_raw[cid] for cid in curr_ids]

        # Build IoU cost matrix (we minimize, so use 1 - IoU)
        cost_matrix = np.ones((len(prev_ids), len(curr_ids)), dtype=np.float64)
        for i, pid in enumerate(prev_ids):
            pm = prev_masks[pid]
            for j, cm in enumerate(curr_masks_list):
                # Resize if dimensions differ (edge case)
                if pm.shape != cm.shape:
                    continue
                intersection = np.logical_and(pm > 0, cm > 0).sum()
                union = np.logical_or(pm > 0, cm > 0).sum()
                if union > 0:
                    iou = intersection / union
                    cost_matrix[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        new_frame: dict[int, np.ndarray] = {}
        assigned_cols: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - cost_matrix[r, c]
            if iou >= iou_threshold:
                # Inherit the previous frame's tracking ID
                inherited_id = prev_ids[r]
                new_frame[inherited_id] = curr_masks_list[c]
                assigned_cols.add(c)
                n_reassigned += 1
            # else: don't inherit — will get a fresh ID below

        # Assign new IDs to unmatched current masks
        for c in range(len(curr_ids)):
            if c not in assigned_cols:
                new_frame[next_id] = curr_masks_list[c]
                next_id += 1

        smoothed[f_idx] = new_frame
        prev_ids = sorted(new_frame.keys())
        prev_masks = new_frame

    logger.info(
        f"IoU temporal smoothing: {n_reassigned} mask IDs inherited "
        f"across {num_frames} frames (IoU threshold={iou_threshold})"
    )

    return smoothed
