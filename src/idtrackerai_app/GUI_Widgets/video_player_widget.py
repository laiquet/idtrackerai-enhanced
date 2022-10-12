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
    QSizePolicy,
)
from time import perf_counter
from PyQt6.QtCore import Qt, QTimer
from functools import lru_cache
import cv2
from matplotlib.pyplot import subplots, rcParams
from idtrackerai.animals_detection.segmentation import _process_frame

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = "Arial"


class MplCanvas:
    def __init__(self):
        self.fig, self.ax = subplots()
        self.fig.patch.set_facecolor("#EFEFEF")
        self.ax.set_facecolor("#EFEFEF")
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        self.ax.set(
            xticks=(), ylabel="Area in pixels", xlabel="Detected blobs"
        )
        self.min_area_line = self.ax.axhline(
            0, linestyle=":", color="gray", visible=False
        )
        self.bars = self.ax.bar([], [])

        self.hide_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarShadeButton
        )
        self.show_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarUnshadeButton
        )

        self.push_btn = QPushButton()
        self.push_btn.setIcon(self.hide_icon)
        self.push_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.push_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.push_btn.clicked.connect(self.show_hide_event)
        self.push_btn.setFixedHeight(15)
        self.bars_visible = True

    def show_hide_event(self):
        self.bars_visible = not self.bars_visible
        self.fig.canvas.setVisible(self.bars_visible)
        if self.bars_visible:
            self.push_btn.setIcon(self.hide_icon)
        else:
            self.push_btn.setIcon(self.show_icon)

    def update(self, number_of_animals, list_of_areas=None):
        if list_of_areas is not None:
            self.areas = list_of_areas
        number_of_blobs = len(self.areas)
        self.bars.remove()
        if number_of_blobs > number_of_animals:
            color = "#BA2320"
            edgecolor = "#5A1010"
            title_prefix = "More blobs than animals! "
        else:
            color = "#44A0D9"
            edgecolor = "#286384"
            title_prefix = ""

        self.bars = self.ax.bar(
            range(number_of_blobs),
            self.areas,
            color=color,
            edgecolor=edgecolor,
            width=0.65,
        )

        if number_of_blobs == 0:
            self.ax.set(title="No blobs detected")
            self.min_area_line.set_visible(False)
            self.ax.set(ylim=(0, 1))
        elif number_of_blobs == 1:
            self.ax.set(
                title=f"1 blob detected of area {self.areas[0]:.0f} px"
            )
            self.min_area_line.set_ydata(self.areas[0])
            self.min_area_line.set_visible(True)
            self.ax.set(ylim=(0, 1.1 * self.areas[0]), xlim=(-0.5, 0.5))
        elif number_of_blobs > 1:
            min_area = min(self.areas)
            self.ax.set(
                title=f"{number_of_blobs} blobs detected. {title_prefix}"
                f"Minimum area: {min_area:.0f} px"
            )
            self.min_area_line.set_ydata(min_area)
            self.min_area_line.set_visible(True)
            self.ax.set(
                ylim=(0, 1.1 * max(self.areas)),
                xlim=(-0.5, number_of_blobs - 0.5),
            )
        else:
            raise TypeError

        self.fig.canvas.draw()


class VideoPlayerWidget(MplFigure):
    def __init__(self, param_func):
        super().__init__()
        self.param_func = param_func
        self.video_holder = VideoHolder()
        self.params = {}

        self.control_bar = QHBoxLayout()

        self.slider_widget = QSlider(Qt.Orientation.Horizontal, minimum=0)
        self.slider_widget.valueChanged.connect(self.sld_changed)

        self.frame_indicator_widget = QSpinBox(minimum=0, value=0)
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

        self.blob_polygons = self.ax.fill()

        self.time_indicator_widget = QLabel()
        self.time_indicator_widget.setFixedHeight(24)

        self.play_pause_button = QPushButton()
        self.play_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_MediaPlay
        )
        self.pause_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause
        )

        self.play_pause_button.setIcon(self.play_icon)
        self.play_pause_button.clicked.connect(self.play_pause_clicked)

        self.control_bar.addWidget(self.play_pause_button)
        self.control_bar.addWidget(self.frame_indicator_widget)
        self.control_bar.addWidget(self.slider_widget)
        self.control_bar.addWidget(self.time_indicator_widget)

        self.area_chart_widget = MplCanvas()
        self.area_chart_widget.fig.canvas.setFocusPolicy(
            Qt.FocusPolicy.NoFocus
        )
        self.VideoPlayer_layout = QVBoxLayout()
        self.VideoPlayer_layout.addWidget(
            self.area_chart_widget.fig.canvas, 30
        )
        self.VideoPlayer_layout.addWidget(self.area_chart_widget.push_btn)
        self.VideoPlayer_layout.addWidget(self.fig.canvas, 62)
        self.VideoPlayer_layout.addLayout(self.control_bar, 8)

        self.current_frame = 0
        self.time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_next_frame)
        self.mask_polygons = []

    def play_pause_clicked(self):
        if not self.fig.canvas.isEnabled():
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_button.setIcon(self.play_icon)
        else:
            self.timer.start()  # 10 fps
            self.play_pause_button.setIcon(self.pause_icon)

    def sld_changed(self):
        self.current_frame = self.slider_widget.value()
        self.frame_indicator_widget.blockSignals(True)
        self.frame_indicator_widget.setValue(self.current_frame)
        self.frame_indicator_widget.blockSignals(False)
        self.update_player()

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

        if isinstance(self.animal_detection_parameters["ROI_mask"], int):
            if self.animal_detection_parameters["ROI_mask"] == 0:
                areas = []
                contours = []
        else:
            (_, _, _, areas, _, contours, _,) = _process_frame(
                frame,
                self.animal_detection_parameters,
                self.current_frame,
                save_pixels="NONE",
                save_segmentation_image="NONE",
            )
        resreduct = self.param_func["resolution_reduction"]()
        if resreduct != 1:
            contours = [contour / resreduct for contour in contours]
        # if animal_detection_parameters["resolution_reduction"] != 1:
        #     frame = cv2.resize(
        #         frame,
        #         None,
        #         fx=animal_detection_parameters["resolution_reduction"],
        #         fy=animal_detection_parameters["resolution_reduction"],
        #         interpolation=cv2.INTER_AREA,
        #     )

        for polygon in self.blob_polygons:
            polygon.remove()

        list_to_fill = []

        for contour in contours:
            list_to_fill.append(contour[..., 0])
            list_to_fill.append(contour[..., 1])
        self.blob_polygons = self.ax.fill(
            *list_to_fill, color="#44A0D9", edgecolor="#286384", lw=1
        )

        self.min_time_between_frames = 1 / self.video_holder.fps
        self.area_chart_widget.update(
            self.param_func["number_of_animals"](), areas
        )
        self.im.set_data(frame)
        self.draw_and_flush()

    def auto_next_frame(self):
        time_between_frames = perf_counter() - self.time
        if time_between_frames < self.min_time_between_frames:
            return
        self.time = perf_counter()
        self.current_frame += 1
        if self.current_frame >= self.video_holder.n_frames:
            self.current_frame = 0
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
        self.video_holder.load(path)
        self.slider_widget.setMaximum(self.video_holder.n_frames - 1)
        self.frame_indicator_widget.setMaximum(self.video_holder.n_frames - 1)
        self.im.set_extent(
            (
                0,
                self.video_holder.size[0],
                self.video_holder.size[1],
                0,
            )
        )
        self.x_center = self.video_holder.size[0] / 2
        self.y_center = self.video_holder.size[1] / 2
        self.fit_zoom(*self.video_holder.size)

        self.current_frame = 0
        self.new_params()

    def update_mask(self, polygons):
        while self.mask_polygons:
            self.mask_polygons.pop().remove()

        for polygon in polygons:
            self.mask_polygons.append(self.ax.add_patch(polygon))
        self.new_params()

    def new_params(self):
        self.animal_detection_parameters = {
            key: value() for key, value in self.param_func.items()
        }

        # TODO is this necessary?
        if not self.animal_detection_parameters["ROI_mask"].any():
            self.animal_detection_parameters["ROI_mask"] = 0

        self.update_player()


class VideoHolder:
    """This class loads the `cv2.VideoCapture` object of the desired
    video path and provides the desired gray-scale frames with
    memoization in `frame(frame_number)`"""

    def __init__(self, path=None):
        if path:
            self.load(path)

    def load(self, path):
        self.path = path
        print(path)
        self.cap = cv2.VideoCapture(str(path))
        self.frame.cache_clear()

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def n_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self):
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @lru_cache(128)
    def frame(self, frame_number):
        if frame_number != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = self.cap.read()
        assert ret
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
