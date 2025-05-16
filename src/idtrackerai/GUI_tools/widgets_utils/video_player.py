# Each Qt binding is different, so...
# pyright: reportIncompatibleMethodOverride=false
import logging
from collections.abc import Iterable, Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np
import toml
from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import QEvent, QPoint, QPointF, QSize, Qt, QThread, QTimer
from qtpy.QtGui import QAction  # type: ignore[reportPrivateImportUsage]
from qtpy.QtGui import (
    QCloseEvent,
    QColor,
    QColorConstants,
    QIcon,
    QImage,
    QKeyEvent,
    QPainter,
    QPixmap,
)
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import GUIBase
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
    painter.drawPolygon(
        (QPoint(0, 0), QPoint(0, size), QPoint(size, size // 2))  # type:ignore
    )
    return canvas


def pause_pixmap(size: int):
    canvas = QPixmap(size, size)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    a = size // 3
    pen = painter.pen()
    pen.setColor(QColor(0x404F40))
    pen.setWidth(2)
    painter.setBrush(QColor(0x809F70))
    painter.setPen(pen)
    painter.drawPolygon(
        (QPoint(0, 0), QPoint(a, 0), QPoint(a, size), QPoint(0, size))  # type:ignore
    )
    painter.drawPolygon(
        (  # type:ignore
            QPoint(size - a, 0),
            QPoint(size, 0),
            QPoint(size, size),
            QPoint(size - a, size),
        )
    )
    return canvas


class AsyncFrameLoader(QThread):
    precomputed_frame = Signal(int, bool, object)

    def __init__(self, holder: VideoPathHolder) -> None:
        super().__init__()
        self.holder = holder
        self.abort = False

    def precompute_frames(self, frame_indices: Iterable[int], color: bool) -> None:
        self.abort = False
        self.frame_indices = frame_indices
        self.color = color
        return super().start()

    def run(self) -> None:
        for frame_index in self.frame_indices:
            if self.abort:
                return
            try:
                frame = self.holder.read_frame(frame_index, self.color)
            except Exception as exc:
                logging.error(f"In {self.__class__.__name__}: {exc}")
            else:
                self.precomputed_frame.emit(frame_index, self.color, frame)


class VideoPlayer(QWidget):
    painting_time = Signal(QPainter, int, object)  # np.ndarray|None
    control_bar_h = 30

    def __init__(self, parent: GUIBase):
        super().__init__(parent)
        self.settings = parent.settings
        self.canvas = Canvas(self)
        self.video_path_holder = VideoPathHolder()

        self.frame_preloader = AsyncFrameLoader(VideoPathHolder())
        self.frame_preloader.precomputed_frame.connect(
            self.video_path_holder.populate_cache
        )

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
        self.forward_loop = QTimer()
        self.backward_loop = QTimer()
        self.forward_loop.setTimerType(Qt.TimerType.PreciseTimer)
        self.backward_loop.setTimerType(Qt.TimerType.PreciseTimer)
        self.forward_loop.timeout.connect(self.forward_step)
        self.backward_loop.timeout.connect(self.backward_step)
        self.fps = 1
        self.drawn_frame = -1
        self.speed: int = 1
        self.speed_label: str = ""
        self.canvas.painting_time.connect(self.paint_video)

        menu = parent.menuBar()
        assert menu is not None
        menu = menu.addMenu("Video player")
        assert menu is not None

        self.draw_in_color = QAction("Enable color", self)
        self.draw_in_color.setCheckable(True)
        menu.addAction(self.draw_in_color)
        self.draw_in_color.toggled.connect(self.update)
        self.draw_in_color.setChecked(
            self.settings.value("video_player/draw_in_color", False, type=bool)
        )

        self.limit_framerate = QAction("Limit framerate", self)
        self.limit_framerate.setShortcut("Ctrl+L")
        self.limit_framerate.setCheckable(True)
        menu.addAction(self.limit_framerate)
        self.limit_framerate.toggled.connect(self.limit_framerate_toggled)
        self.limit_framerate.setChecked(
            self.settings.value("video_player/limit_framerate", True, type=bool)
        )

        self.reduce_cache = QAction("Reduce memory usage", self)
        self.reduce_cache.setCheckable(True)
        menu.addAction(self.reduce_cache)
        self.reduce_cache.toggled.connect(self.video_path_holder.set_cache_mode)
        self.reduce_cache.setChecked(
            self.settings.value("video_player/reduce_cache", False, type=bool)
        )

        playback_speed_action = QAction("Change playback speed", self)
        playback_speed_action.triggered.connect(lambda: ChangePlaybackSpeed(self, self.speed))  # type: ignore
        menu.addAction(playback_speed_action)

        tooltips = toml.load(Path(__file__).parent.parent / "tooltips.toml")
        self.draw_in_color.setToolTip(tooltips["color_action"])
        self.limit_framerate.setToolTip(tooltips["framerate_action"])
        self.reduce_cache.setToolTip(tooltips["reducecache_action"])
        menu.setToolTipsVisible(True)
        self.limit_framerate.setChecked(True)
        parent.installEventFilter(self)

        # We draw the video at (-0.5, -0.5) so that the pixel coordinates are in their center
        self.video_drawing_origin = QPointF(-0.5, -0.5)
        self.forward_is_pressed = False
        self.backward_is_pressed = False

        self.key_hold_delay_timer = QTimer()
        self.key_hold_delay_timer.setSingleShot(True)
        self.key_hold_delay_timer.timeout.connect(self.keyHeld)

    def limit_framerate_toggled(self, is_framerate_limited: bool):
        interval = int(1000 / self.fps) if is_framerate_limited else 0
        self.forward_loop.setInterval(interval)
        self.backward_loop.setInterval(interval)

    def preload_frames(self, start: int, end: int):
        """Preloads the frames in the video_path_holder cache"""
        color = self.draw_in_color.isChecked()
        if self.frame_preloader.isRunning():
            self.frame_preloader.abort = True
            self.frame_preloader.wait()

        with suppress(RuntimeError):
            self.frame_preloader.precompute_frames(range(start, end), color)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        play_btn_size = self.frame_indicator.height()
        play_icon_size = int(play_btn_size * 0.6 + 0.5)
        self.play_pause_button.setFixedSize(play_btn_size, play_btn_size)
        self.play_pause_button.setIconSize(QSize(play_icon_size, play_icon_size))

    def closeEvent(self, event: QCloseEvent):
        self.settings.setValue(
            "video_player/reduce_cache", self.reduce_cache.isChecked()
        )
        self.settings.setValue(
            "video_player/draw_in_color", self.draw_in_color.isChecked()
        )
        self.settings.setValue(
            "video_player/limit_framerate", self.limit_framerate.isChecked()
        )
        self.video_path_holder.clear_cache()
        super().closeEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowBlocked:
            self.stop_all()
        return super().event(event)

    def stop_all(self):
        self.play_pause_button.setChecked(False)
        self.forward_loop.stop()
        self.backward_loop.stop()

    def play_pause_clicked(self, play: bool):
        self.forward_loop.stop()
        self.backward_loop.stop()
        if play:
            self.forward_loop.start()
        else:
            self.forward_loop.stop()

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
        hours = seconds // 3600
        return f"{hours:02d}:{minutes:02d}:{seconds % 60:02d}"

    def paint_video(self, painter: QPainter) -> None:
        if not self.isEnabled():
            return

        current_frame = self.current_frame
        self.time_indicator_widget.setText(self.current_time)
        color = self.draw_in_color.isChecked()

        try:
            # TODO run in a thread and display info in case it takes too long
            frame = self.video_path_holder.frame(current_frame, color)
        except RuntimeError as exc:  # unreadable frame by OpenCV
            if self.drawn_frame != current_frame:
                logging.error(exc)  # avoid printing multiple equal logs
            painter.fillRect(
                0, 0, self.video_width, self.video_height, QColorConstants.DarkGray
            )
            painter.setPen(QColorConstants.White)
            painter.drawText(
                painter.window(),
                Qt.AlignmentFlag.AlignCenter,
                str(exc).replace(" of ", " of\n"),
            )
            self.painting_time.emit(painter, current_frame, None)
        else:
            painter.drawPixmap(
                self.video_drawing_origin,
                QPixmap.fromImage(
                    QImage(
                        frame.data,
                        frame.shape[1],
                        frame.shape[0],
                        frame.shape[1] * 3 if color else frame.shape[1],
                        (
                            QImage.Format.Format_BGR888
                            if color
                            else QImage.Format.Format_Grayscale8
                        ),
                    )
                ),
            )
            self.painting_time.emit(painter, current_frame, frame)

        if self.speed_label:
            painter.resetTransform()
            painter.setFont(self.font())
            painter.setPen(QColorConstants.White)
            painter.drawText(
                self.canvas.rect(), Qt.AlignmentFlag.AlignBottom, self.speed_label
            )

        self.drawn_frame = current_frame

    def backward_step(self) -> None:
        new_frame = self.current_frame - self.speed
        if new_frame < 0:
            new_frame = self.n_frames
        self.frame_indicator.setValue(new_frame)

    def forward_step(self) -> None:
        new_frame = self.current_frame + self.speed
        if new_frame >= self.n_frames:
            new_frame = 0
        self.frame_indicator.setValue(new_frame)

    def eventFilter(self, object, event: QEvent) -> bool:
        """Catch key events even when VideoPlayer is not in focus."""
        if event.type() == QEvent.Type.KeyPress:
            assert isinstance(event, QKeyEvent)
            return self.keyPressEvent_from_eventFilter(event)
        if event.type() == QEvent.Type.KeyRelease:
            assert isinstance(event, QKeyEvent)
            return self.keyReleaseEvent_from_eventFilter(event)
        return False  # keep processing the event

    def keyHeld(self) -> None:
        if self.forward_is_pressed:
            self.forward_loop.start()
        if self.backward_is_pressed:
            self.backward_loop.start()

    def keyPressEvent_from_eventFilter(self, event: QKeyEvent) -> bool:
        if (
            event.isAutoRepeat()
            or not self.isEnabled()
            or event.modifiers()
            not in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier)
        ):
            return False
        key = event.key()
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.play_pause_button.setChecked(False)
            self.forward_step()
            self.forward_is_pressed = True
            self.key_hold_delay_timer.start(200)
            return True
        if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.play_pause_button.setChecked(False)
            self.backward_step()
            self.backward_is_pressed = True
            self.key_hold_delay_timer.start(200)
            return True
        with suppress(ValueError):
            self.setSpeed(int(event.text()))
            return True
        return False

    def keyReleaseEvent_from_eventFilter(self, event: QKeyEvent) -> bool:
        if event.isAutoRepeat():
            return False
        key = event.key()
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.forward_is_pressed = False
            self.forward_loop.stop()
            self.key_hold_delay_timer.stop()
            return True
        if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.backward_is_pressed = False
            self.backward_loop.stop()
            self.key_hold_delay_timer.stop()
            return True
        return False

    def setSpeed(self, value: int):
        if value == 0:
            return
        self.speed = 2 ** (value - 1)
        self.speed_label = f"Speed x{self.speed}" if self.speed != 1 else ""
        self.update()

    def update_video_paths(
        self,
        video_paths: Sequence[Path],
        n_frames: int,
        video_size: tuple[int, int],
        fps: float,
    ) -> None:
        self.fps = fps
        self.limit_framerate_toggled(self.limit_framerate.isChecked())
        self.n_frames = n_frames
        self.video_width, self.video_height = video_size
        self.video_path_holder.load_paths(video_paths)
        self.frame_preloader.holder.load_paths(video_paths)
        self.frame_slider.setMaximum(n_frames - 1)
        self.frame_indicator.setMaximum(n_frames - 1)
        self.frame_indicator.setValue(0)
        self.canvas.adjust_zoom_to(video_size[0], video_size[1])
        self.update()

    def reorder_video_paths(self, video_paths: Sequence[Path]) -> None:
        self.video_path_holder.load_paths(video_paths)
        self.frame_preloader.holder.load_paths(video_paths)
        self.update()

    def center_canvas_at(self, x: float, y: float, zoom_scale: float) -> None:
        self.canvas.zoom = zoom_scale / min(self.width(), self.height())
        self.canvas.centerX = x
        self.canvas.centerY = y


class ChangePlaybackSpeed(QDialog):
    def __init__(self, parent: VideoPlayer, current: int) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setLayout(QVBoxLayout())
        self.slider = QSlider(Qt.Orientation.Horizontal)
        layout = self.layout()
        assert layout is not None
        layout.addWidget(self.slider)
        layout.addWidget(QLabel("Change the speed by pressing a numeric key"))
        self.slider.setMinimum(1)
        self.slider.setMaximum(9)
        self.slider.setValue(int(np.log2(current)) + 1)
        self.slider.valueChanged.connect(parent.setSpeed)
        self.exec()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        with suppress(ValueError):
            self.slider.setValue(int(event.text()))
