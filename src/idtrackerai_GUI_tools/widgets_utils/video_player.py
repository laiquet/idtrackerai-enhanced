import json
from contextlib import suppress
from pathlib import Path
from time import perf_counter

import numpy as np
import toml
from qtpy.QtCore import QEvent, QRectF, QSize, Qt, QTimer, Signal  # type: ignore
from qtpy.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QColorConstants,
    QIcon,
    QImage,
    QKeyEvent,
    QPainter,
    QPixmap,
    QPolygon,
)
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .canvas import Canvas
from .video_paths_holder import VideoPathHolder


def play_pixmap(size: int):
    canvas = QPixmap(size, size)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    pen = painter.pen()
    pen.setColor(QColor(0x306F00))
    pen.setWidth(2)
    painter.setBrush(QColor(0xC0DF50))
    painter.setPen(pen)
    poly = QPolygon()
    poly.setPoints(0, 0, 0, size, size, size // 2)
    painter.drawPolygon(poly)
    return canvas


def pause_pixmap(size: int):
    canvas = QPixmap(size, size)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    a = size // 3
    poly = QPolygon()
    pen = painter.pen()
    pen.setColor(QColor(0x404F40))
    pen.setWidth(2)
    painter.setBrush(QColor(0x809F70))
    painter.setPen(pen)
    poly.setPoints(0, 0, a, 0, a, size, 0, size)
    painter.drawPolygon(poly)
    poly.setPoints(size - a, 0, size, 0, size, size, size - a, size)
    painter.drawPolygon(poly)
    return canvas


class VideoPlayer(QWidget):
    painting_time = Signal(QPainter, int, np.ndarray)
    control_bar_h = 30

    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.canvas = Canvas(self)
        self.video_path_holder = VideoPathHolder()

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_indicator = QSpinBox()

        self.frame_slider.valueChanged.connect(self.frame_indicator.setValue)
        self.frame_slider.sliderPressed.connect(self.stop_all)

        self.frame_indicator.valueChanged.connect(self.frame_indicator_changed)
        self.frame_indicator.setKeyboardTracking(False)
        self.frame_indicator.editingFinished.connect(self.frame_indicator.clearFocus)

        self.time_indicator_widget = QLabel()
        self.play_pause_button = QToolButton()
        self.play_pause_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_pause_button.setShortcut(Qt.Key.Key_Space)
        self.play_pause_button.setCheckable(True)

        icon = QIcon()
        icon.addPixmap(play_pixmap(60), QIcon.Mode.Normal, QIcon.State.Off)
        icon.addPixmap(pause_pixmap(60), QIcon.Mode.Normal, QIcon.State.On)

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
        self.forward_timer = QTimer()
        self.backward_timer = QTimer()
        self.play_loop.timeout.connect(self.forward_loop)
        self.forward_timer.timeout.connect(self.forward_loop)
        self.backward_timer.timeout.connect(self.backward_loop)
        self.min_time_between_frames = 1
        self.fps = 1
        self.drawn_frame = -1
        self.speed: int = 1
        self.speed_label: str = ""
        self.speed_label_timer = QTimer()
        self.speed_label_timer.setSingleShot(True)
        self.speed_label_timer.setInterval(3000)
        self.speed_label_timer.timeout.connect(self.removeSpeedlabel)

        self.freeze = False
        self.canvas.painting_time.connect(self.paint_video)

        menu = parent.menuBar().addMenu("Video player")

        self.draw_in_color = QAction("Enable color", self)
        self.draw_in_color.setCheckable(True)
        menu.addAction(self.draw_in_color)
        self.draw_in_color.toggled.connect(self.update)

        self.limit_framerate = QAction("Limit framerate", self)
        self.limit_framerate.setShortcut("Ctrl+L")
        self.limit_framerate.setCheckable(True)
        menu.addAction(self.limit_framerate)

        self.reduce_cache = QAction("Reduce memory usage", self)
        self.reduce_cache.setCheckable(True)
        menu.addAction(self.reduce_cache)
        self.VideoPlayer_param_path = Path(__file__).parent / "video_player.json"
        self.reduce_cache.toggled.connect(self.video_path_holder.set_cache_mode)
        self.reduce_cache.setChecked(
            json.load(self.VideoPlayer_param_path.open())["reduce_cache"]
            if self.VideoPlayer_param_path.is_file()
            else False
        )

        playback_speed_action = QAction("Change playback speed", self)
        playback_speed_action.triggered.connect(lambda: ChangePlaybackSpeed(self, self.speed))  # type: ignore
        menu.addAction(playback_speed_action)

        def limit_framerate_toggled(state: bool):
            self.min_time_between_frames = 1 / self.fps if state else 0

        self.limit_framerate.toggled.connect(limit_framerate_toggled)

        tooltips = toml.load((Path(__file__).parent.parent / "tooltips.toml"))
        self.draw_in_color.setToolTip(tooltips["color_action"])
        self.limit_framerate.setToolTip(tooltips["framerate_action"])
        self.reduce_cache.setToolTip(tooltips["reducecache_action"])
        menu.setToolTipsVisible(True)
        parent.installEventFilter(self)

    def preload_frames(self, start: int, end: int):
        """Preloads the frames in the video_path_holder cache"""
        color = self.draw_in_color.isChecked()
        for frame in range(start, end):
            self.video_path_holder.frame(frame, color)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        play_btn_size = self.frame_indicator.height()
        play_icon_size = int(play_btn_size * 0.6 + 0.5)
        self.play_pause_button.setFixedSize(play_btn_size, play_btn_size)
        self.play_pause_button.setIconSize(QSize(play_icon_size, play_icon_size))

    def closeEvent(self, event: QCloseEvent):
        json.dump(
            {"reduce_cache": self.reduce_cache.isChecked()},
            self.VideoPlayer_param_path.open("w"),
        )
        super().closeEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowBlocked:
            self.stop_all()
        return super().event(event)

    def stop_all(self):
        self.play_pause_button.setChecked(False)
        self.forward_timer.stop()
        self.backward_timer.stop()

    def play_pause_clicked(self, play: bool):
        self.forward_timer.stop()
        self.backward_timer.stop()
        if play:
            self.play_loop.start()
        else:
            self.play_loop.stop()

    def frame_indicator_changed(self, frame_indicator_value):
        self.frame_slider.setValue(frame_indicator_value)
        self.update()

    def setCurrentFrame(self, frame, force_update=False):
        self.stop_all()
        if force_update and self.frame_indicator.value() == frame:
            self.update()
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
        color = self.draw_in_color.isChecked()
        frame = self.video_path_holder.frame(current_frame, color)

        painter.drawImage(
            self.rect_to_draw_image,
            QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                (frame.shape[1] * 3 if color else frame.shape[1]),
                (
                    QImage.Format.Format_BGR888
                    if color
                    else QImage.Format.Format_Grayscale8
                ),
            ),
        )
        self.painting_time.emit(painter, current_frame, frame)

        painter.resetTransform()
        painter.setFont(self.font())
        painter.setPen(QColorConstants.White)
        if self.speed_label:
            painter.drawText(
                self.canvas.rect(), Qt.AlignmentFlag.AlignBottom, self.speed_label
            )

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

    def backward_loop(self):
        if self.pass_frame():
            return
        new_frame = self.current_frame - self.speed
        if new_frame < 0:
            new_frame = self.n_frames
        self.frame_indicator.setValue(new_frame)

    def forward_loop(self):
        if self.pass_frame():
            return
        new_frame = self.current_frame + self.speed
        if new_frame >= self.n_frames:
            new_frame = 0
        self.frame_indicator.setValue(new_frame)

    def eventFilter(self, object, event: QEvent) -> bool:
        """Catch key events even when VideoPlayer is not in focus."""
        if event.type() == QEvent.Type.KeyPress:
            self.keyPressEvent(event)  # type: ignore
        if event.type() == QEvent.Type.KeyRelease:
            self.keyReleaseEvent(event)  # type: ignore
        return False  # keep processing the event

    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.freeze = True
            self.forward_timer.start()
            self.play_pause_button.setChecked(False)
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.freeze = True
            self.backward_timer.start()
            self.play_pause_button.setChecked(False)
        with suppress(ValueError):
            self.setSpeed(int(event.text()))
        event.ignore()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.forward_timer.stop()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.backward_timer.stop()

    def setSpeed(self, value: int):
        if value == 0:
            return
        self.speed = 2 ** (value - 1)
        self.speed_label = f"Speed x{self.speed}"
        self.speed_label_timer.start()
        self.update()

    def removeSpeedlabel(self):
        self.speed_label = ""
        self.update()

    def update_video_paths(
        self, video_paths, n_frames, video_size, fps, res_reduct=1.0
    ):
        self.fps = fps
        self.min_time_between_frames = (
            1 / fps if self.limit_framerate.isChecked() else 0
        )
        self.n_frames = n_frames
        self.video_width, self.video_height = video_size
        self.video_path_holder.load_paths(video_paths)
        self.frame_slider.setMaximum(n_frames - 1)
        self.frame_indicator.setMaximum(n_frames - 1)
        self.frame_indicator.setValue(0)
        self.canvas.adjust_zoom_to(
            res_reduct * video_size[0], res_reduct * video_size[1]
        )
        self.set_resolution_reduction(res_reduct)
        self.update()

    def set_resolution_reduction(self, value: float):
        if not hasattr(self, "video_height"):
            return
        self.rect_to_draw_image = QRectF(
            -0.5, -0.5, value * self.video_width, value * self.video_height
        )

    def reorder_video_paths(self, video_paths):
        self.video_path_holder.load_paths(video_paths)
        self.update()

    def center_canvas_at(self, x: float, y: float, zoom_scale: float):
        self.canvas.zoom = zoom_scale / min(self.width(), self.height())
        self.canvas.centerX = x
        self.canvas.centerY = y


class ChangePlaybackSpeed(QDialog):
    def __init__(self, parent: VideoPlayer, current: int):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setLayout(QVBoxLayout())
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(QLabel("Change the speed by pressing a numeric key"))
        self.slider.setMinimum(1)
        self.slider.setMaximum(9)
        self.slider.setValue(int(np.log2(current)) + 1)
        self.slider.valueChanged.connect(parent.setSpeed)
        self.exec()

    def keyPressEvent(self, event: QKeyEvent):
        with suppress(ValueError):
            self.slider.setValue(int(event.text()))
