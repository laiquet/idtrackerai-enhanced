from idtrackerai_app.widgets_utils import MplCanvas, VideoPathHolder_Cls
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QSpinBox,
    QSlider,
    QStyle,
    QCommonStyle,
    QWidget,
)
from time import perf_counter
from PyQt6.QtCore import Qt, QTimer, pyqtSignal


class VideoPlayer(QWidget):
    frame_ready_to_draw = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.canvas = MplCanvas()
        self.VideoPathHolder = VideoPathHolder_Cls()

        self.frame_slider = QSlider(Qt.Orientation.Horizontal, minimum=0)
        self.frame_slider.valueChanged.connect(self.sld_changed)

        self.frame_indicator = QSpinBox(minimum=0, value=0)
        self.frame_indicator.valueChanged.connect(self.frame_indicator_changed)
        self.frame_indicator.setKeyboardTracking(False)
        self.frame_indicator.editingFinished.connect(
            lambda: self.frame_indicator.clearFocus()
        )

        self.im = self.canvas.ax.imshow(
            [[]],
            cmap="gray",
            vmax=255,
            vmin=0,
            extent=[0, 1, 1, 0],
            interpolation="none",
            animated=True,
            resample=False,
            snap=False,
        )

        self.time_indicator_widget = QLabel()
        self.play_pause_button = QPushButton()
        self.play_pause_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_MediaPlay
        )
        self.pause_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause
        )

        self.play_pause_button.setIcon(self.play_icon)
        self.play_pause_button.clicked.connect(self.play_pause_clicked)
        self.frame_indicator.setFixedHeight(30)
        self.time_indicator_widget.setFixedHeight(30)
        self.play_pause_button.setFixedSize(30, 30)
        self.frame_slider.setFixedHeight(30)

        self.control_bar = QHBoxLayout()
        self.control_bar.addWidget(self.play_pause_button)
        self.control_bar.addWidget(self.frame_indicator)
        self.control_bar.addWidget(self.frame_slider)
        self.control_bar.addWidget(self.time_indicator_widget)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)
        self.layout().addLayout(self.control_bar)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.time = 0
        self.play_loop = QTimer()
        self.forward_loop = QTimer()
        self.backward_loop = QTimer()
        self.play_loop.timeout.connect(self.next_frame)
        self.forward_loop.timeout.connect(self.next_frame)
        self.backward_loop.timeout.connect(self.previous_frame)
        self.min_time_between_frames = 1
        self.fps = 1
        self.drawn_frame = -1

    def play_pause_clicked(self):
        self.forward_loop.stop()
        self.backward_loop.stop()
        if self.play_loop.isActive():
            self.play_loop.stop()
            self.play_pause_button.setIcon(self.play_icon)
        else:
            self.play_loop.start()
            self.play_pause_button.setIcon(self.pause_icon)

    def sld_changed(self, sld_value):
        self.frame_indicator.setValue(sld_value)

    def frame_indicator_changed(self, frame_indicator_value):
        self.frame_slider.setValue(frame_indicator_value)
        self.update_player()

    def setCurrentFrame(self, frame):
        self.frame_indicator.setValue(frame)

    @property
    def current_frame(self):
        return self.frame_indicator.value()

    @property
    def current_time(self) -> str:
        seconds = int(self.current_frame / self.fps)
        minutes = (seconds // 60) % 60
        hours = (seconds // 3600) % 60
        return f"{hours:02d}:{minutes:02d}:{seconds%60:02d}"

    def update_player(self):
        current_frame = self.current_frame
        self.time_indicator_widget.setText(self.current_time)
        if current_frame != self.drawn_frame:
            self.im.set_data(self.VideoPathHolder.frame(current_frame))
        self.frame_ready_to_draw.emit()
        self.canvas.draw_and_flush()
        self.drawn_frame = current_frame

    def pass_frame(self):
        if (perf_counter() - self.time) < self.min_time_between_frames:
            return True
        else:
            self.time = perf_counter()
            return False

    def previous_frame(self, check_fps=True):
        if check_fps:
            if self.pass_frame():
                return
        new_frame = max(0, self.current_frame - 1)
        self.frame_indicator.setValue(new_frame)

    def next_frame(self, check_fps=True):
        if check_fps:
            if self.pass_frame():
                return
        new_frame = min(self.n_frames - 1, self.current_frame + 1)
        self.frame_indicator.setValue(new_frame)

    def redirect_keyPressEvent(self, key):
        if key == " ":
            self.play_pause_clicked()
            return
        elif key in ("d", "right"):
            self.next_frame(check_fps=False)
            self.time = perf_counter() + 0.2
            self.forward_loop.start()
        elif key in ("a", "left"):
            self.previous_frame(check_fps=False)
            self.time = perf_counter() + 0.2
            self.backward_loop.start()
        if self.play_loop.isActive():
            self.play_pause_clicked()

    def redirect_keyReleaseEvent(self, key):
        if key in ("d", "right"):
            self.forward_loop.stop()
            self.time = 0
        elif key in ("a", "left"):
            self.backward_loop.stop()
            self.time = 0

    def update_video_paths(self, video_paths, n_frames, video_size, fps):
        self.fps = fps
        self.min_time_between_frames = 1 / fps
        self.n_frames = n_frames
        self.VideoPathHolder.load_paths(video_paths)
        self.frame_slider.setMaximum(n_frames - 1)
        self.frame_indicator.setMaximum(n_frames - 1)
        self.im.set_extent((0, *video_size, 0))
        self.canvas.x_center = video_size[0] / 2
        self.canvas.y_center = video_size[1] / 2
        self.canvas.fit_zoom(*video_size)
        self.update_player()

    def reorder_video_paths(self, video_paths):
        self.VideoPathHolder.load_paths(video_paths)
        self.update_player()
