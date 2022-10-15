from idtrackerai_app.widgets_utils import MplFigure
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QSpinBox,
    QSlider,
    QStyle,
    QCommonStyle,
)
from time import perf_counter
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from idtrackerai.animals_detection.segmentation import _process_frame
import cv2
from functools import lru_cache


class VideoPlayerWidget(QVBoxLayout):
    new_areas = pyqtSignal(int, list)

    def __init__(self, params):
        super().__init__()
        self.plot = MplFigure()
        self.VideoPathHolder = VideoPathHolder_Cls()
        self.params = params

        self.slider_widget = QSlider(Qt.Orientation.Horizontal, minimum=0)
        self.slider_widget.valueChanged.connect(self.sld_changed)

        self.frame_indicator = QSpinBox(minimum=0, value=0)
        self.frame_indicator.valueChanged.connect(self.frame_indicator_changed)
        self.frame_indicator.setKeyboardTracking(False)
        self.frame_indicator.editingFinished.connect(
            lambda: self.frame_indicator.clearFocus()
        )

        self.im = self.plot.ax.imshow(
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

        self.blob_polygons = self.plot.ax.fill()

        self.time_indicator_widget = QLabel()

        self.play_pause_button = QPushButton()
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
        self.slider_widget.setFixedHeight(30)

        self.control_bar = QHBoxLayout()
        self.control_bar.addWidget(self.play_pause_button)
        self.control_bar.addWidget(self.frame_indicator)
        self.control_bar.addWidget(self.slider_widget)
        self.control_bar.addWidget(self.time_indicator_widget)

        self.addWidget(self.plot.fig.canvas)
        self.addLayout(self.control_bar)

        self.time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_next_frame)
        self.mask_polygons = []

    def play_pause_clicked(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_button.setIcon(self.play_icon)
        else:
            self.timer.start()
            self.play_pause_button.setIcon(self.pause_icon)

    def sld_changed(self, sld_value):
        self.frame_indicator.setValue(sld_value)

    def frame_indicator_changed(self, frame_indicator_value):
        self.slider_widget.setValue(frame_indicator_value)
        self.update_player()

    def setCurrentFrame(self, frame):
        self.frame_indicator.setValue(frame)

    def setEnabled(self, enabled):
        self.enabled = enabled

    @property
    def current_frame(self):
        return self.frame_indicator.value()

    def update_player(self):
        current_frame = self.current_frame
        seconds = int(current_frame / self.params["video_fps"]())
        minutes = (seconds // 60) % 60
        hours = (seconds // 3600) % 60

        self.time_indicator_widget.setText(
            f"{hours:02d}:{minutes:02d}:{seconds% 60:02d}"
        )

        frame = self.VideoPathHolder.frame(current_frame)

        if isinstance(self.animal_detection_parameters["ROI_mask"], int):
            if self.animal_detection_parameters["ROI_mask"] == 0:
                areas = []
                contours = []
        else:
            (_, _, _, areas, _, contours, _,) = _process_frame(
                frame,
                self.animal_detection_parameters,
                current_frame,
                save_pixels="NONE",
                save_segmentation_image="NONE",
            )

        resreduct = self.params["resolution_reduction"]()
        if resreduct != 1:
            contours = [contour / resreduct for contour in contours]

        for polygon in self.blob_polygons:
            polygon.remove()

        list_to_fill = []

        for contour in contours:
            list_to_fill.append(contour[..., 0])
            list_to_fill.append(contour[..., 1])
        self.blob_polygons = self.plot.ax.fill(
            *list_to_fill,
            color="#44A0D9",
            edgecolor="#286384",
            lw=1,
        )

        self.min_time_between_frames = 1 / self.params["video_fps"]()
        self.new_areas.emit(current_frame, areas)
        self.im.set_data(frame)
        self.plot.draw_and_flush()

    def auto_next_frame(self):
        time_between_frames = perf_counter() - self.time
        if time_between_frames < self.min_time_between_frames:
            return
        self.time = perf_counter()
        new_frame = self.current_frame + 1
        if new_frame >= self.params["video_n_frames"]():
            new_frame = 0
        self.frame_indicator.setValue(new_frame)

    def redirect_keyPressEvent(self, key):
        if key in ("d", "right"):
            self.frame_indicator.setValue(
                min(
                    self.params["video_n_frames"]() - 1, self.current_frame + 1
                )
            )
        elif key in ("a", "left"):
            self.frame_indicator.setValue(max(0, self.current_frame - 1))
        elif key == " ":
            self.play_pause_clicked()

    def update_video_paths(self, video_paths):
        self.VideoPathHolder.load_paths(video_paths)
        self.slider_widget.setMaximum(self.params["video_n_frames"]() - 1)
        self.frame_indicator.setMaximum(self.params["video_n_frames"]() - 1)
        self.im.set_extent(
            (
                0,
                *self.params["video_size"](),
                0,
            )
        )
        self.plot.x_center = self.params["video_size"]()[0] / 2
        self.plot.y_center = self.params["video_size"]()[1] / 2
        self.plot.fit_zoom(*self.params["video_size"]())

        self.new_params()

    def update_mask(self, polygons):
        while self.mask_polygons:
            self.mask_polygons.pop().remove()

        for polygon in polygons:
            self.mask_polygons.append(self.plot.ax.add_patch(polygon))
        self.new_params()

    def reorder_video_paths(self, video_paths):
        self.VideoPathHolder.load_paths(video_paths)
        self.update_player()

    def new_params(self):
        self.animal_detection_parameters = {
            key: value() for key, value in self.params.items()
        }

        if not self.animal_detection_parameters["ROI_mask"].any():
            self.animal_detection_parameters["ROI_mask"] = 0

        self.update_player()


class VideoPathHolder_Cls:
    def load_paths(self, video_paths):
        assert video_paths
        self.single_file = len(video_paths) == 1
        self.interval_dict = {}
        i = 0

        for video_path in video_paths:
            n_frames = int(
                cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
            )
            self.interval_dict[video_path] = (i, i + n_frames)
            i += n_frames
        self.cap = cv2.VideoCapture(video_paths[0])
        self.current_captured_video_path = video_paths[0]
        self.frame.cache_clear()

    @lru_cache(128)
    def frame(self, frame_number):

        for path, (start, end) in self.interval_dict.items():
            if frame_number >= start and frame_number < end:
                break

        if path != self.current_captured_video_path:
            self.cap = cv2.VideoCapture(str(path))
            self.current_captured_video_path = path

        frame_number_in_path = frame_number - start

        if frame_number_in_path != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_in_path)
        ret, img = self.cap.read()
        assert (
            ret
        ), f"Error on frame {frame_number}, {frame_number_in_path} of {str(path)}"
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
