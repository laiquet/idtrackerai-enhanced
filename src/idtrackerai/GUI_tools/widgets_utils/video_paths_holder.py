from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


class VideoPathHolder:
    cap: cv2.VideoCapture
    current_captured_video_path: Path | None

    def __init__(self, video_paths: list[Path] | None = None) -> None:
        self.video_loaded = False
        self.reduced_cache = False
        if video_paths:
            self.load_paths(video_paths)

    def load_paths(self, video_paths: list[Path]) -> None:
        assert video_paths
        self.single_file = len(video_paths) == 1
        self.interval_dict: dict[Path, tuple[int, int]] = {}
        i = 0

        for video_path in video_paths:
            n_frames = int(
                cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT)
            )
            self.interval_dict[video_path] = (i, i + n_frames)
            i += n_frames
        self.current_captured_video_path = None
        self.clear_cache()
        self.video_loaded = True

    def clear_cache(self) -> None:
        self.frame_large_cache.cache_clear()
        self.frame_small_cache.cache_clear()

    def set_cache_mode(self, reduced: bool) -> None:
        self.reduced_cache = reduced
        self.clear_cache()

    def frame(self, frame_number: int, color: bool) -> np.ndarray:
        if self.reduced_cache:
            return self.frame_small_cache(frame_number, color)
        return self.frame_large_cache(frame_number, color)

    # TODO check flake8 warnings
    @lru_cache(128)  # noqa: B019
    def frame_large_cache(self, frame_number: int, color: bool) -> np.ndarray:
        return self.read_frame(frame_number, color)

    @lru_cache(16)  # noqa: B019
    def frame_small_cache(self, frame_number: int, color: bool) -> np.ndarray:
        return self.read_frame(frame_number, color)

    def read_frame(self, frame_number: int, color: bool) -> np.ndarray:
        if not self.video_loaded:
            return np.array([[]])
        for path_i, (start, end) in self.interval_dict.items():
            if start <= frame_number < end:
                path = path_i
                break
        else:
            raise ValueError(
                f"Frame number {frame_number} not in intervals {self.interval_dict}"
            )

        if path != self.current_captured_video_path:
            self.cap = cv2.VideoCapture(str(path))
            self.cap.read()  # this somehow initializes the caption and makes following actions less prone to errors, magically
            self.current_captured_video_path = path

        frame_number_in_path = frame_number - start

        if frame_number_in_path != self.cap.get(cv2.CAP_PROP_POS_FRAMES):
            success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_in_path)
            if (
                not success
                or self.cap.get(cv2.CAP_PROP_POS_FRAMES) != frame_number_in_path
            ):
                raise RuntimeError(
                    f"OpenCV could not jump to frame {frame_number}"
                    + (
                        f", {frame_number_in_path}"
                        if len(self.interval_dict) > 1
                        else ""
                    )
                    + f" of {path}"
                )
        ret, img = self.cap.read()
        if not ret:
            raise RuntimeError(
                f"OpenCV could not read frame {frame_number}"
                + (f", {frame_number_in_path}" if len(self.interval_dict) > 1 else "")
                + f" of {path}"
            )

        if color:
            return img  # BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
