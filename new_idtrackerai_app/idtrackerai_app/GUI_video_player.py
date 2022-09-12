from matplotlib_widget import matplotlib_gui
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QSpinBox,
    QSlider,
    QStyle,
)

from PyQt6.QtCore import Qt, QTimer
from functools import lru_cache
import cv2
from matplotlib.pyplot import subplots


class MplCanvas:
    def __init__(self):
        self.fig, self.ax = subplots()
        self.fig.patch.set_facecolor("#EFEFEF")
        self.ax.set_facecolor("#EFEFEF")
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        self.canvas = self.fig.canvas
        self.bars = self.ax.bar([], [])

    def update(self, list_of_areas):
        self.bars.remove()
        self.bars = self.ax.bar(
            range(len(list_of_areas)),
            list_of_areas,
            color="#44A0D9",
            edgecolor="#286384",
        )
        self.ax.relim()
        self.fig.canvas.draw()


class VideoPlayer(QWidget, matplotlib_gui):
    def __init__(self, video_path=None, actual_conf=None):
        super().__init__()
        self.canvas.setEnabled(False)
        self.video_holder = VideoHolder(video_path)

        self.control_bar = QHBoxLayout()

        self.slider_widget = QSlider(
            Qt.Orientation.Horizontal, minimum=0, enabled=False
        )
        self.slider_widget.valueChanged.connect(self.sld_changed)

        self.frame_indicator_widget = QSpinBox(
            enabled=False, minimum=0, value=0
        )
        self.frame_indicator_widget.valueChanged.connect(
            self.frame_indicator_changed
        )
        self.frame_indicator_widget.setKeyboardTracking(False)
        self.frame_indicator_widget.editingFinished.connect(
            lambda: self.frame_indicator_widget.clearFocus()
        )

        self.im = self.ax.imshow(
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

        self.polygons = self.ax.fill()

        self.time_indicator_widget = QLabel()
        self.time_indicator_widget.setFixedHeight(24)

        self.play_pause_button = QPushButton(enabled=False)
        self.play_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPlay
        )
        self.pause_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause
        )

        self.play_pause_button.setIcon(self.play_icon)
        self.play_pause_button.clicked.connect(self.play_pause_clicked)

        self.control_bar.addWidget(self.play_pause_button)
        self.control_bar.addWidget(self.frame_indicator_widget)
        self.control_bar.addWidget(self.slider_widget)
        self.control_bar.addWidget(self.time_indicator_widget)

        # self.setCentral
        # self.canvas.setFocusPolicy(Qt.StrongFocus)

        # self.zoom = 1
        # self.set_ax_lims()

        self.area_chart_widget = MplCanvas()
        self.area_chart_widget.canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.area_chart_widget.canvas.setVisible(False)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.area_chart_widget.canvas, 30)
        self.main_layout.addWidget(self.canvas, 62)
        self.main_layout.addLayout(self.control_bar, 8)

        self.current_frame = 0
        # self.update_player()

        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_next_frame)

    def play_pause_clicked(self):
        if not self.canvas.isEnabled():
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_button.setIcon(self.play_icon)
        else:
            self.timer.start()
            self.play_pause_button.setIcon(self.pause_icon)

    # @pyqtSlot()
    def sld_changed(self):
        self.current_frame = self.slider_widget.value()
        self.frame_indicator_widget.blockSignals(True)
        self.frame_indicator_widget.setValue(self.current_frame)
        self.frame_indicator_widget.blockSignals(False)
        self.update_player()

    # @pyqtSlot()
    def frame_indicator_changed(self):
        self.current_frame = self.frame_indicator_widget.value()
        self.slider_widget.blockSignals(True)
        self.slider_widget.setValue(self.current_frame)
        self.slider_widget.blockSignals(False)
        self.update_player()

    def update_player(self):
        seconds = int(self.current_frame / self.video_holder.fps)
        minutes = (seconds // 60) % 60
        hours = (seconds // 3600) % 60

        self.time_indicator_widget.setText(
            f"{hours:02d}:{minutes:02d}:{seconds% 60:02d}"
        )

        frame = self.video_holder.frame(self.current_frame)

        ret, thresh = cv2.threshold(frame, 145, 255, cv2.THRESH_BINARY)
        out = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = out[0] if len(out) == 2 else out[1]

        for polygon in self.polygons:
            polygon.remove()

        list_to_fill = []
        list_of_areas = []

        for contour in contours[1:]:
            list_to_fill.append(contour[..., 0])
            list_to_fill.append(contour[..., 1])
            list_to_fill.append("r")
            list_of_areas.append(cv2.contourArea(contour))
        self.polygons = self.ax.fill(*list_to_fill)

        self.area_chart_widget.update(list_of_areas)
        self.im.set_data(frame)
        self.draw_and_flush()

    def auto_next_frame(self):
        self.current_frame = min(
            self.video_holder.n_frames - 1, self.current_frame + 1
        )
        self.frame_indicator_widget.setValue(self.current_frame)

    def redirect_keyPressEvent(self, key):
        if key == "d":
            self.current_frame = min(
                self.video_holder.n_frames - 1, self.current_frame + 1
            )
            self.frame_indicator_widget.setValue(self.current_frame)
        elif key == "a":
            self.current_frame = max(0, self.current_frame - 1)
            self.frame_indicator_widget.setValue(self.current_frame)
        elif key == " ":
            self.play_pause_clicked()

    def update_video(self, path):
        enable = path is not None
        self.slider_widget.setEnabled(enable)
        self.frame_indicator_widget.setEnabled(enable)
        self.canvas.setEnabled(enable)
        self.slider_widget.setEnabled(enable)
        self.play_pause_button.setEnabled(enable)

        self.video_holder.load(path)
        self.slider_widget.setMaximum(self.video_holder.n_frames - 1)
        self.frame_indicator_widget.setMaximum(self.video_holder.n_frames - 1)
        self.im.set_extent(
            (
                0,
                self.video_holder.width,
                self.video_holder.height,
                0,
            )
        )
        self.x_center = self.video_holder.width / 2
        self.y_center = self.video_holder.height / 2
        self.set_ax_lims()

        self.current_frame = 0
        self.update_player()


class VideoHolder:
    """This class loads the `cv2.VideoCapture` object of the desired video path and provides the desired gray-scale frames with memoization in `frame(frame_number)`"""

    def __init__(self, path=None):
        if path:
            self.load(path)

    def load(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame.cache_clear()

    @lru_cache(128)
    def frame(self, frame_number):
        if frame_number != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = self.cap.read()
        assert ret
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
