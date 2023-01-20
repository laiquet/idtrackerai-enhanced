from time import perf_counter

import numpy as np
from idtrackerai_app.widgets_utils import Canvas, VideoPathHolder
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCommonStyle,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class VideoPlayer(QWidget):
    painting_time = pyqtSignal(QPainter, int, np.ndarray)

    def __init__(self):
        super().__init__()
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

        self.im = QPixmap()  # , QImage()

        self.time_indicator_widget = QLabel()
        self.play_pause_button = QPushButton()
        self.play_pause_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_icon = QCommonStyle().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
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

    def stop_all(self):
        if self.play_loop.isActive():
            self.play_loop.stop()
            self.play_pause_button.setIcon(self.play_icon)
        self.forward_loop.stop()
        self.backward_loop.stop()

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

        frame = self.VideoPathHolder.frame(current_frame)
        # im = QImage(
        #     frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_Grayscale8
        # )

        # imQt = QImage(ImageQt.ImageQt(im8))
        # a = QByteArray(frame.size, b"\x05")
        # print(a[0])
        # npa = np.frombuffer(memoryview(a), dtype=np.uint8)
        # npa[:] = frame.ravel()
        # print(a[0])
        pxmap = QPixmap.fromImage(
            QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                QImage.Format.Format_Grayscale8,
            )
        )
        # print(pxmap.size(), frame.size)
        # print(pxmap.loadFromData(str(frame.tobytes())))
        # self.im = QPixmap(frame))
        painter.drawPixmap(self.rect_to_draw_image, pxmap, QRectF(pxmap.rect()))

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

        print(f"  {1/elapsed_time:.4f} fps", end="\r")
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
        if key == Qt.Key.Key_Space:
            self.play_pause_clicked()
            return
        elif key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.freeze = True
            self.forward_loop.start()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.freeze = True
            self.backward_loop.start()
        if self.play_loop.isActive():
            self.play_pause_clicked()

    def redirect_keyReleaseEvent(self, key):
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.forward_loop.stop()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.backward_loop.stop()

    def update_video_paths(
        self, video_paths, n_frames, video_size, fps, res_reduct=1.0
    ):
        self.fps = fps
        self.min_time_between_frames = 1 / 50  # 1 / fps
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
