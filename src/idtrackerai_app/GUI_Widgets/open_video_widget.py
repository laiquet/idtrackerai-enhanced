from PyQt6.QtWidgets import (
    QLabel,
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
        print(video_paths)
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
                self._video_width,
                self._video_height,
                fps,
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

        self.video_paths = video_paths
        self.new_video_paths.emit()

    def video_height(self):
        return self._video_height

    def video_width(self):
        return self._video_width

    def get_video_paths(self):
        return self.video_paths

    def getEpisodes(self):
        return self.episodes

    def setEnabled(self, enabled):
        self.button_open.setEnabled(enabled)
