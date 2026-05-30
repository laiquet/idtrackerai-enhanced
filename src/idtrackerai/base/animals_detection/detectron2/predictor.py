"""Detectron2 instance segmentation predictor for idtrackerai.

This module wraps Detectron2's ``DefaultPredictor`` to provide
per-frame animal instance segmentation from a pretrained model.

The output format is identical to the SAM 3 predictor:
``{frame_idx: {obj_id: binary_mask}}``, so all downstream processing
(mask-to-blob conversion, overlap linking, fragmentation, etc.)
works without modification.

Uses Hungarian matching on centroids between consecutive frames
to maintain consistent object IDs (same strategy as SAM 3 per-frame
fallback).
"""

import gc
import logging
import os

import cv2
import numpy as np
import torch

logger = logging.getLogger("idtrackerai.detectron2.predictor")


def _free_gpu():
    """Release GPU memory after inference."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class Detectron2AnimalSegmenter:
    """Detectron2-based animal instance segmenter.

    All parameters are **mandatory**. Users must provide their own
    config YAML, pretrained weights, and target class name(s).

    Parameters
    ----------
    config_path : str
        Path to a Detectron2 config YAML file.
    weights_path : str
        Path to model weights (``.pth`` / ``.pkl``).
    confidence_threshold : float
        Score threshold for instance detections.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        if not config_path:
            raise ValueError(
                "detectron2_config is required. "
                "Provide the path to your Detectron2 config YAML file."
            )
        if not weights_path:
            raise ValueError(
                "detectron2_weights is required. "
                "Provide the path to your pretrained model weights."
            )
        self.config_path = config_path
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device

    def segment_video(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        roi_mask: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ) -> dict[int, dict[int, np.ndarray]]:
        """Run Detectron2 instance segmentation on a video.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        start_frame : int
            Global start frame index.
        end_frame : int
            Global end frame index (exclusive).
        roi_mask : np.ndarray or None
            Binary ROI mask. Pixels outside the ROI are zeroed.
        class_names : list[str]
            Class names to filter detections. Must match the class names
            in the model's training dataset. Required.

        Returns
        -------
        video_segments : dict
            ``{frame_idx: {obj_id: binary_mask_array}}``
            where ``frame_idx`` is relative (0-based from start_frame).
        """
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError:
            raise ImportError(
                "Detectron2 is required for the 'detectron2' segmentation method.\n"
                "Install it with: pip install idtrackerai[detectron2]\n"
                "Or follow: https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
            )

        # Build config
        cfg = get_cfg()
        cfg.merge_from_file(self.config_path)
        cfg.MODEL.WEIGHTS = self.weights_path

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.DEVICE = self.device if torch.cuda.is_available() else "cpu"

        logger.info(f"Detectron2 config: {self.config_path}")
        logger.info(f"Detectron2 weights: {cfg.MODEL.WEIGHTS}")
        logger.info(f"Detectron2 device: {cfg.MODEL.DEVICE}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")

        predictor = DefaultPredictor(cfg)

        # Resolve class ID filter from class_names (mandatory).
        #
        # For standard datasets (COCO, LVIS): resolve names via MetadataCatalog.
        # For custom models: the model may not have registered metadata.
        #   - If model has only 1 class → all detections are that class, no
        #     filtering needed.
        #   - If class_names are integers → use as direct class IDs.
        #   - Otherwise → skip filtering (the model was trained specifically
        #     for these objects, so all its detections are relevant).
        if not class_names:
            raise ValueError(
                "detectron2_class_names is required. "
                "Provide the class name(s) your model was trained on, "
                "e.g. ['fish'] or ['animal']."
            )

        class_id_filter: set[int] | None = None
        thing_classes: list[str] = []

        # Step 1: Try to resolve from registered dataset metadata
        try:
            from detectron2.data import MetadataCatalog
            try:
                import detectron2.data.datasets  # noqa: F401
            except Exception:
                pass
            dataset_name = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else ""
            if dataset_name:
                meta = MetadataCatalog.get(dataset_name)
                thing_classes = meta.get("thing_classes", [])
        except Exception:
            pass

        if thing_classes:
            # Metadata available — match by name (case-insensitive)
            name_set = {n.lower().strip() for n in class_names}
            class_id_filter = {
                i for i, name in enumerate(thing_classes)
                if name.lower() in name_set
            }
            if class_id_filter:
                matched = [thing_classes[i] for i in sorted(class_id_filter)]
                logger.info(
                    f"Filtering to classes: {matched} "
                    f"(IDs: {sorted(class_id_filter)})"
                )
            else:
                logger.warning(
                    f"Class names {class_names} not found in dataset "
                    f"metadata ({thing_classes}). Will try integer IDs."
                )
                # Fall through to step 2

        # Step 2: Try interpreting as integer class IDs
        if class_id_filter is None:
            try:
                class_id_filter = {int(n) for n in class_names}
                logger.info(
                    f"Using class_names as integer class IDs: "
                    f"{sorted(class_id_filter)}"
                )
            except ValueError:
                pass  # Not integers — fall through to step 3

        # Step 3: Custom model without registered metadata
        # The user trained this model for their specific objects.
        # All detections from the model are what they want.
        if class_id_filter is None:
            num_model_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            logger.info(
                f"Target classes: {class_names} "
                f"(model has {num_model_classes} class(es)). "
                f"No dataset metadata registered — keeping all detections "
                f"from the model."
            )
            # class_id_filter stays None → no filtering applied

        # Process video frames
        num_frames = end_frame - start_frame
        video_segments: dict[int, dict[int, np.ndarray]] = {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # Hungarian matching state for consistent IDs
        prev_centroids: dict[int, tuple[float, float]] = {}
        next_available_id = 1

        for frame_idx in range(num_frames):
            global_frame = start_frame + frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Could not read frame {global_frame}")
                video_segments[frame_idx] = {}
                continue

            # Apply ROI mask
            if roi_mask is not None:
                frame = frame.copy()
                frame[roi_mask == 0] = 0

            # Run Detectron2 inference
            try:
                outputs = predictor(frame)
            except Exception as exc:
                logger.warning(f"Frame {frame_idx}: inference failed ({exc})")
                video_segments[frame_idx] = {}
                continue

            instances = outputs["instances"].to("cpu")

            # Filter by class if specified
            if class_id_filter is not None and len(instances) > 0:
                keep = torch.tensor([
                    int(c) in class_id_filter
                    for c in instances.pred_classes
                ])
                instances = instances[keep]

            # Extract binary masks
            raw_masks: list[np.ndarray] = []
            if len(instances) > 0 and instances.has("pred_masks"):
                masks_tensor = instances.pred_masks.numpy()
                for i in range(len(instances)):
                    binary_mask = masks_tensor[i].astype(np.uint8)
                    raw_masks.append(binary_mask)

            # Hungarian matching for consistent IDs
            frame_masks: dict[int, np.ndarray] = {}

            if raw_masks:
                # Compute centroids
                curr_centroids: list[tuple[float, float]] = []
                for mask in raw_masks:
                    ys, xs = np.where(mask > 0)
                    if len(ys) > 0:
                        curr_centroids.append(
                            (float(ys.mean()), float(xs.mean()))
                        )
                    else:
                        curr_centroids.append((0.0, 0.0))

                if prev_centroids and curr_centroids:
                    prev_ids = list(prev_centroids.keys())
                    prev_pts = [prev_centroids[pid] for pid in prev_ids]

                    # Build cost matrix
                    cost = np.zeros(
                        (len(prev_pts), len(curr_centroids)),
                        dtype=np.float64,
                    )
                    for i, (py, px) in enumerate(prev_pts):
                        for j, (cy, cx) in enumerate(curr_centroids):
                            cost[i, j] = np.sqrt(
                                (py - cy) ** 2 + (px - cx) ** 2
                            )

                    try:
                        from scipy.optimize import linear_sum_assignment

                        row_ind, col_ind = linear_sum_assignment(cost)
                        assigned_cols: set[int] = set()
                        new_centroids: dict[int, tuple[float, float]] = {}

                        for r, c in zip(row_ind, col_ind):
                            obj_id = prev_ids[r]
                            frame_masks[obj_id] = raw_masks[c]
                            new_centroids[obj_id] = curr_centroids[c]
                            assigned_cols.add(c)

                        for c in range(len(raw_masks)):
                            if c not in assigned_cols:
                                frame_masks[next_available_id] = raw_masks[c]
                                new_centroids[next_available_id] = (
                                    curr_centroids[c]
                                )
                                next_available_id += 1

                        prev_centroids = new_centroids

                    except ImportError:
                        logger.warning(
                            "scipy not available for Hungarian matching, "
                            "using positional IDs"
                        )
                        prev_centroids = {}
                        for idx, mask in enumerate(raw_masks):
                            oid = idx + 1
                            frame_masks[oid] = mask
                            prev_centroids[oid] = curr_centroids[idx]
                else:
                    # First frame
                    prev_centroids = {}
                    for idx, mask in enumerate(raw_masks):
                        obj_id = next_available_id
                        frame_masks[obj_id] = mask
                        prev_centroids[obj_id] = curr_centroids[idx]
                        next_available_id += 1

            video_segments[frame_idx] = frame_masks

            if frame_idx % 100 == 0 and frame_idx > 0:
                logger.info(
                    f"Processed {frame_idx}/{num_frames} frames"
                )

        cap.release()
        del predictor
        _free_gpu()

        # Log summary
        total_detections = sum(
            len(masks) for masks in video_segments.values()
        )
        avg_per_frame = (
            total_detections / num_frames if num_frames > 0 else 0
        )
        logger.info(
            f"Detectron2 segmentation complete: {total_detections} detections "
            f"across {num_frames} frames ({avg_per_frame:.1f} avg/frame)"
        )

        return video_segments

    def release(self) -> None:
        """Release GPU resources."""
        _free_gpu()
        logger.info("Detectron2 predictor released from GPU")
