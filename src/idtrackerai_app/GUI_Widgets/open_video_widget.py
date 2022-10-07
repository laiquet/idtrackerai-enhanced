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
import os
from PyQt6.QtCore import Qt, pyqtSignal
import cv2
from idtrackerai.video import Video


class OpenBtnWidget(QHBoxLayout):
    new_video_loaded = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.button_open = QPushButton("Open")
        self.button_open.clicked.connect(self.button_open_clicked)
        self.button_open.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button_open.setFixedHeight(28)
        self.button_open.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )
        self.QLabel = QLabel("Video:")
        self.layout().addWidget(self.QLabel)
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
                self.parent,
                "Open a video file to track",
                filter="Video (*.avi *.mp4 *.mpg *.mov *.AVI *.MP4 *.MPG *.MOV);; All (*)",
            )
        if fileName:
            # TODO get better text adaptation (at resize)
            if len(fileName) > 50:
                self.button_open.setText(
                    os.path.join("...", os.path.split(fileName)[-1])
                )
            else:
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
            self.new_video_loaded.emit()

    def video_height(self):
        return self._video_height

    def video_width(self):
        return self._video_width

    def video_paths(self):
        return self._video_paths

    def episodes(self):
        return self._episodes

    def setEnabled(self, enabled):
        self.QLabel.setEnabled(enabled)
        self.button_open.setEnabled(enabled)
