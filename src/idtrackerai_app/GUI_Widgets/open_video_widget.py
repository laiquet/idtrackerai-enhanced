from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QSizePolicy,
    QStyle,
)
from PyQt6.QtCore import Qt
import cv2
from idtrackerai.video import Video


class OpenBtnWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.button_open = QPushButton("Open")
        self.button_open.clicked.connect(self.button_open_clicked)
        self.button_open.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button_open.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )

        self.layout().addWidget(QLabel("Video:"))
        self.layout().addWidget(self.button_open)
        self._video_height = None
        self._video_width = None
        self._video_paths = None
        self._episodes = 54

    def button_open_clicked(self, opened=None):
        if opened:
            fileName = opened
        else:
            fileName, _ = QFileDialog.getOpenFileName(
                self,
                "Open a video file to track",
                filter="Video (*.avi *.mp4 *.mpg *.mov *.AVI *.MP4 *.MPG *.MOV);; All (*)",
            )
        if fileName:
            self.button_open.setText(fileName)

            multiple_files = False

            self._video_paths = Video.get_video_paths(fileName, multiple_files)
            (
                self.video_paths_n_frames,
                self._tracking_intervals,
                self._episodes,
            ) = Video.get_processing_episodes([fileName], None)

            cap = cv2.VideoCapture(self._video_paths[0])
            self._video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def video_height(self):
        return self._video_height

    def video_width(self):
        return self._video_width

    def video_paths(self):
        return self._video_paths

    def episodes(self):
        return self._episodes
