import cv2
from functools import lru_cache


class VideoPathHolder_Cls:
    def load_paths(self, video_paths):
        assert video_paths
        self.single_file = len(video_paths) == 1
        self.interval_dict = {}
        i = 0

        for video_path in video_paths:
            n_frames = int(
                cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
            )
            self.interval_dict[video_path] = (i, i + n_frames)
            i += n_frames
        self.cap = cv2.VideoCapture(video_paths[0])
        self.current_captured_video_path = video_paths[0]
        self.frame.cache_clear()

    @lru_cache(128)
    def frame(self, frame_number):

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
        ), f"Error on frame {frame_number}, {frame_number_in_path} of {str(path)}"
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
