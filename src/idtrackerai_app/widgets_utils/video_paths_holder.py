from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


class VideoPathHolder:
    def __init__(self, video_paths=None):
        self.video_loaded = False
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
        self.cap = cv2.VideoCapture(str(video_paths[0]))
        self.current_captured_video_path = video_paths[0]
        self.frame.cache_clear()
        self.video_loaded = True

    @lru_cache(128)
    def frame(self, frame_number: int) -> np.ndarray:

        if not self.video_loaded:
            return np.array([[]])
        for path, (start, end) in self.interval_dict.items():
            if frame_number >= start and frame_number < end:
                break

        if path != self.current_captured_video_path:
            self.cap = cv2.VideoCapture(str(path))
            self.current_captured_video_path = path

        frame_number_in_path = frame_number - start

        if frame_number_in_path != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_in_path)
        ret, img = self.cap.read()
        assert (
            ret
        ), f"Error on frame {frame_number}, {frame_number_in_path} of {path}"
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
