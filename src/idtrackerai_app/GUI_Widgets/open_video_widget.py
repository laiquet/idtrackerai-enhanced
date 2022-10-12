from PyQt6.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QListWidget,
    QListView,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from idtrackerai.video import Video
from confapp import conf
from idtrackerai_app.widgets_utils import MessageBox, WrappedLabel
import cv2
from functools import lru_cache


class OpenVideoWidget(QHBoxLayout):
    new_video_paths = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.avaliable_extensions = conf.AVAILABLE_VIDEO_EXTENSION
        self.extension_filter = (
            "Video (*" + " *".join(self.avaliable_extensions) + ");; All (*)"
        )
        self.parent = parent
        self.button_open = QPushButton("Open video")
        self.button_open.clicked.connect(self.button_open_clicked)
        self.button_open.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button_open.setFixedHeight(28)
        self.button_open.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.list_of_files = QListWidget()
        # TODO remove focus of multiple file list
        # self.list_of_files.setMaximumHeight(100)
        # self.list_of_files.setSizeAdjustPolicy()
        self.list_of_files.setDefaultDropAction(Qt.MoveAction)
        self.list_of_files.setMovement(QListView.Free)
        self.single_file_label = WrappedLabel()
        self.layout().addWidget(self.button_open)
        self.layout().addWidget(self.list_of_files)
        self.layout().addWidget(self.single_file_label)
        self.list_of_files.setVisible(False)
        self.single_file_label.setVisible(False)
        self.wrong_input_popup = MessageBox(parent, title="Wrong video paths")

    def button_open_clicked(self, checked=False, video_paths=None):
        if video_paths is None:
            video_paths, _ = QFileDialog.getOpenFileNames(
                self.parent,
                "Open a video file to track",
                filter=self.extension_filter,
            )

        if not video_paths:
            return

        try:
            video_paths = Video.process_video_paths(video_paths)
            (
                self.video_width,
                self.video_height,
                self.fps,
            ) = Video.get_info_from_video_paths(video_paths)
        except (ValueError, AssertionError) as e:
            self.wrong_input_popup.exec(str(e))
            return

        if len(video_paths) == 1:
            self.single_file_label.setText(str(video_paths[0]))
            self.single_file_label.setVisible(True)
            self.list_of_files.setVisible(False)

        else:
            self.list_of_files.clear()
            self.list_of_files.addItems([str(path) for path in video_paths])
            self.single_file_label.setVisible(False)
            self.list_of_files.setVisible(True)

        n_rows = min(5, len(video_paths)) + 1
        self.list_of_files.setFixedHeight(
            self.list_of_files.sizeHintForRow(0) * n_rows
            + 2 * self.list_of_files.frameWidth(),
        )

        (
            self.video_paths_n_frames,
            _,
            self.episodes,
        ) = Video.get_processing_episodes(video_paths)

        self.n_frames = sum(self.video_paths_n_frames)

        self.video_paths = video_paths
        self.cap = cv2.VideoCapture(str(video_paths[0]))
        self.current_open_path = video_paths[0]
        self.frame.cache_clear()
        self.new_video_paths.emit()

    @lru_cache(128)
    def frame(self, frame_number):

        start_frame = -self.video_paths_n_frames[0]
        end_frame = 0

        for i in range(len(self.video_paths)):
            start_frame += self.video_paths_n_frames[i]
            end_frame += self.video_paths_n_frames[i]
            if frame_number < end_frame and frame_number >= start_frame:
                path = self.video_paths[i]
                break
        if path != self.current_open_path:
            self.cap = cv2.VideoCapture(str(path))

        frame_number_in_path = frame_number - start_frame

        if frame_number_in_path != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_in_path)
        ret, img = self.cap.read()
        assert (
            ret
        ), f"Error on frame {frame_number}, {frame_number_in_path} of {str(path)}"
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def getNframes(self):
        return self.n_frames

    def getVideoPaths(self):
        return self.video_paths

    def getSize(self):
        return self.video_width, self.video_height

    def getEpisodes(self):
        return self.episodes

    def getFps(self):
        return self.fps

    def setEnabled(self, enabled):
        self.button_open.setEnabled(enabled)
