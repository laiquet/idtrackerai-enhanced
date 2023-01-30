from time import perf_counter

import numpy as np
from idtrackerai_app.widgets_utils import Canvas, VideoPathHolder
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QImage, QPainter
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class VideoPlayer(QWidget):
    painting_time = pyqtSignal(QPainter, int, np.ndarray)
    control_bar_h = 30

    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.canvas = Canvas(self)
        self.VideoPathHolder = VideoPathHolder()

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.sld_changed)
        self.frame_slider.sliderPressed.connect(self.stop_all)

        self.frame_indicator = QSpinBox()
        self.frame_indicator.valueChanged.connect(self.frame_indicator_changed)
        self.frame_indicator.setKeyboardTracking(False)
        self.frame_indicator.editingFinished.connect(
            lambda: self.frame_indicator.clearFocus()
        )

        self.time_indicator_widget = QLabel()
        self.play_pause_button = QToolButton()
        self.play_pause_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_pause_button.setShortcut(Qt.Key.Key_Space)
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.setFixedSize(self.control_bar_h, self.control_bar_h)

        icon = QIcon()
        icon.addPixmap(
            self.style().standardPixmap(self.style().StandardPixmap.SP_MediaPlay),
            QIcon.Mode.Normal,
            QIcon.State.Off,
        )
        icon.addPixmap(
            self.style().standardPixmap(self.style().StandardPixmap.SP_MediaPause),
            QIcon.Mode.Normal,
            QIcon.State.On,
        )

        self.play_pause_button.setIcon(icon)
        self.play_pause_button.toggled.connect(self.play_pause_clicked)
        self.time_indicator_widget.setFixedHeight(self.control_bar_h)
        self.frame_slider.setFixedHeight(self.control_bar_h)

        self.control_bar = QHBoxLayout()
        self.control_bar.addWidget(self.play_pause_button)
        self.control_bar.addWidget(self.frame_indicator)
        self.control_bar.addWidget(self.frame_slider)
        self.control_bar.addWidget(self.time_indicator_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(self.control_bar)
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
        self.freeze = False
        self.canvas.painting_time.connect(self.paint_video)

        menu = parent.menuBar().addMenu("Video player")

        self.draw_in_color = QAction("Enable color", self)
        self.draw_in_color.setCheckable(True)
        menu.addAction(self.draw_in_color)
        self.draw_in_color.toggled.connect(self.update)

        self.limit_framerate = QAction("Limit framerate", self)
        self.limit_framerate.setCheckable(True)
        menu.addAction(self.limit_framerate)

        def limit_framerate_toggled(state: bool):
            self.min_time_between_frames = 1 / self.fps if state else 0

        self.limit_framerate.toggled.connect(limit_framerate_toggled)

    def stop_all(self):
        self.play_pause_button.setChecked(False)
        self.forward_loop.stop()
        self.backward_loop.stop()

    def play_pause_clicked(self, play: bool):
        self.forward_loop.stop()
        self.backward_loop.stop()
        if play:
            self.play_loop.start()
        else:
            self.play_loop.stop()

    def sld_changed(self, sld_value):
        self.frame_indicator.setValue(sld_value)

    def frame_indicator_changed(self, frame_indicator_value):
        self.frame_slider.setValue(frame_indicator_value)
        self.update()

    def setCurrentFrame(self, frame):
        self.frame_indicator.setValue(frame)

    @property
    def current_frame(self) -> int:
        return self.frame_indicator.value()

    @property
    def current_time(self) -> str:
        seconds = int(self.current_frame / self.fps)
        minutes = (seconds // 60) % 60
        hours = (seconds // 3600) % 60
        return f"{hours:02d}:{minutes:02d}:{seconds%60:02d}"

    def paint_video(self, painter: QPainter):
        if not self.isEnabled():
            return

        current_frame = self.current_frame
        self.time_indicator_widget.setText(self.current_time)
        if self.draw_in_color.isChecked():
            frame = self.VideoPathHolder.frameColor(current_frame)
            img = QImage(
                frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888
            )
        else:
            frame = self.VideoPathHolder.frame(current_frame)
            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                QImage.Format.Format_Grayscale8,
            )

        painter.drawImage(self.rect_to_draw_image, img)
        # TODO send gray image to signal (maybe is faster?)
        self.painting_time.emit(painter, current_frame, frame)
        self.drawn_frame = current_frame

    def pass_frame(self):
        if not self.isEnabled():
            return True
        if self.freeze:
            self.time = perf_counter() + 0.2
            self.freeze = False
            return False
        elapsed_time = perf_counter() - self.time
        if elapsed_time < self.min_time_between_frames:
            return True

        # print(f"  {1/elapsed_time:.4f} fps", end="\r")
        self.time = perf_counter()
        return False

    def previous_frame(self):
        if self.pass_frame():
            return
        new_frame = max(0, self.current_frame - 1)
        self.frame_indicator.setValue(new_frame)

    def next_frame(self):
        if self.pass_frame():
            return
        new_frame = self.current_frame + 1
        if new_frame == self.n_frames:
            new_frame = 0
        self.frame_indicator.setValue(new_frame)

    def redirect_keyPressEvent(self, key: int):
        self.play_pause_button.setChecked(False)
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.freeze = True
            self.forward_loop.start()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.freeze = True
            self.backward_loop.start()

    def redirect_keyReleaseEvent(self, key):
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.forward_loop.stop()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.backward_loop.stop()

    def update_video_paths(
        self, video_paths, n_frames, video_size, fps, res_reduct=1.0
    ):
        self.fps = fps
        self.min_time_between_frames = (
            1 / fps if self.limit_framerate.isChecked() else 0
        )
        self.n_frames = n_frames
        self.video_width, self.video_height = video_size
        self.VideoPathHolder.load_paths(video_paths)
        self.frame_slider.setMaximum(n_frames - 1)
        self.frame_indicator.setMaximum(n_frames - 1)
        self.frame_indicator.setValue(0)
        self.canvas.adjust_zoom_to(*video_size)
        self.set_resolution_reduction(res_reduct)
        self.update()

    def set_resolution_reduction(self, value: float):
        if not hasattr(self, "video_height"):
            return
        self.rect_to_draw_image = QRectF(
            -0.5, -0.5, value * self.video_width, value * self.video_height
        )

    def reorder_video_paths(self, video_paths):
        self.VideoPathHolder.load_paths(video_paths)
        self.update()
