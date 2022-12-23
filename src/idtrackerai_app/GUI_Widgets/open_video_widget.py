from idtrackerai_app.widgets_utils import MessageBox, WrappedLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListView,
    QListWidget,
    QPushButton,
    QSizePolicy,
)

from idtrackerai import Video
from idtrackerai.utils import conf


class OpenVideoWidget(QHBoxLayout):
    new_video_paths = pyqtSignal(list, tuple, int, int, list)
    path_clicked = pyqtSignal(int)
    video_paths_reordered = pyqtSignal(list)
    pause_video = pyqtSignal()
    new_episodes = pyqtSignal(list, object)

    def __init__(self, parent=None):
        super().__init__()
        self.parent_widget = parent
        self.avaliable_extensions = conf.AVAILABLE_VIDEO_EXTENSION
        self.extension_filter = (
            "Video (*" + " *".join(self.avaliable_extensions) + ");; All (*)"
        )
        self.button_open = QPushButton("Open video")
        self.button_open.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button_open.clicked.connect(self.button_open_clicked)
        self.button_open.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Fixed,
        )
        self.list_of_files = QListWidget()
        self.list_of_files.setAlternatingRowColors(True)
        self.list_of_files.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list_of_files.setDefaultDropAction(Qt.MoveAction)
        self.list_of_files.setMovement(QListView.Free)
        self.list_of_files.model().rowsMoved.connect(
            self.video_paths_reordered_func
        )
        self.single_file_label = WrappedLabel()
        self.layout().addWidget(self.button_open)
        self.layout().addWidget(self.list_of_files)
        self.layout().addWidget(self.single_file_label)
        self.list_of_files.setVisible(False)
        self.list_of_files.itemClicked.connect(self.video_path_clicked)
        self.single_file_label.setVisible(False)
        self.wrong_input_popup = MessageBox(parent, title="Wrong video paths")
        self.tracking_intervals = None

    def video_path_clicked(self, item):
        self.path_clicked.emit(self.video_path_start[item.text()][0])

    def video_paths_reordered_func(self):
        self.video_path_start.clear()
        i = 0
        for video_path in self.video_paths:
            n_frames = self.video_path_n_frames[video_path]
            self.video_path_start[video_path] = (i, i + n_frames)
            i += n_frames
        self.video_paths_reordered.emit(self.video_paths)
        (
            self.n_frames,
            video_paths_n_frames,
            _,
            self.episodes,
        ) = Video.get_processing_episodes(
            self.video_paths, self.tracking_intervals
        )
        self.new_episodes.emit(self.episodes)

    def button_open_clicked(self):
        self.pause_video.emit()
        video_paths, _ = QFileDialog.getOpenFileNames(
            self.parent_widget,
            "Open a video file to track",
            filter=self.extension_filter,
        )
        self.open_video_paths(sorted(video_paths))

    def open_video_paths(self, video_paths):
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

        self.single_file = len(video_paths) == 1
        if self.single_file:
            self.single_file_label.setText(str(video_paths[0]))
        else:
            self.list_of_files.clear()
            self.list_of_files.addItems(map(str, video_paths[::-1]))
            n_rows = min(5, len(video_paths)) + 1
            self.list_of_files.setFixedHeight(
                self.list_of_files.sizeHintForRow(0) * n_rows
                + 2 * self.list_of_files.frameWidth(),
            )

        self.single_file_label.setVisible(self.single_file)
        self.list_of_files.setVisible(not self.single_file)

        (
            self.n_frames,
            video_paths_n_frames,
            _,
            self.episodes,
        ) = Video.get_processing_episodes(video_paths, self.tracking_intervals)
        self.video_path_n_frames = dict(
            zip(self.video_paths, video_paths_n_frames)
        )

        self.video_path_start = {}
        i = 0
        for video_path in self.video_paths:
            n_frames = self.video_path_n_frames[video_path]
            self.video_path_start[video_path] = (i, i + n_frames)
            i += n_frames

        self.n_frames = i
        self.new_video_paths.emit(
            self.video_paths,
            (self.video_width, self.video_height),
            self.n_frames,
            self.fps,
            self.episodes,
        )

    def set_tracking_interval(self, tracking_intervals):
        self.tracking_intervals = tracking_intervals

        (
            self.n_frames,
            video_paths_n_frames,
            _,
            self.episodes,
        ) = Video.get_processing_episodes(
            self.video_paths, self.tracking_intervals
        )
        self.new_episodes.emit(self.video_paths, self.episodes)

    @property
    def video_paths(self):
        return self.getVideoPaths()

    def getNframes(self):
        return self.n_frames

    def getVideoPaths(self) -> list[str]:
        if self.single_file:
            return [self.single_file_label.text()]
        else:
            return [
                self.list_of_files.item(i).text()
                for i in range(self.list_of_files.count())
            ]

    def getSize(self):
        return self.video_width, self.video_height

    def getEpisodes(self):
        return self.episodes

    def getFps(self):
        return self.fps
