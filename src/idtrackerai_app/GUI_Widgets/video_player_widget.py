from idtrackerai_app.widgets_utils import MplFigure, VideoPathHolder
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
        self.areas = []

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
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.control_bar = QHBoxLayout()

        self.slider_widget = QSlider(Qt.Orientation.Horizontal, minimum=0)
        self.slider_widget.valueChanged.connect(self.sld_changed)

        self.frame_indicator = QSpinBox(minimum=0, value=0)
        self.frame_indicator.valueChanged.connect(self.frame_indicator_changed)
        self.frame_indicator.setKeyboardTracking(False)
        self.frame_indicator.editingFinished.connect(
            lambda: self.frame_indicator.clearFocus()
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
        self.control_bar.addWidget(self.frame_indicator)
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
            self.timer.start()
            self.play_pause_button.setIcon(self.pause_icon)

    def sld_changed(self, sld_value):
        self.frame_indicator.setValue(sld_value)

    def frame_indicator_changed(self, frame_indicator_value):
        self.slider_widget.setValue(frame_indicator_value)
        self.update_player()

    def setCurrentFrame(self, frame):
        self.frame_indicator.setValue(frame)

    @property
    def current_frame(self):
        return self.frame_indicator.value()

    def update_player(self):
        print("updating with", self.current_frame)
        seconds = int(self.current_frame / self.params["video_fps"]())
        minutes = (seconds // 60) % 60
        hours = (seconds // 3600) % 60

        self.time_indicator_widget.setText(
            f"{hours:02d}:{minutes:02d}:{seconds% 60:02d}"
        )

        frame = VideoPathHolder.frame(self.current_frame)

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
        resreduct = self.params["resolution_reduction"]()
        if resreduct != 1:
            contours = [contour / resreduct for contour in contours]

        for polygon in self.blob_polygons:
            polygon.remove()

        list_to_fill = []

        for contour in contours:
            list_to_fill.append(contour[..., 0])
            list_to_fill.append(contour[..., 1])
        self.blob_polygons = self.ax.fill(
            *list_to_fill, color="#44A0D9", edgecolor="#286384", lw=1
        )

        self.min_time_between_frames = 1 / self.params["video_fps"]()
        self.area_chart_widget.update(
            self.params["number_of_animals"](), areas
        )
        self.im.set_data(frame)
        self.draw_and_flush()

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
        if key == "d":
            self.frame_indicator.setValue(
                min(
                    self.params["video_n_frames"]() - 1, self.current_frame + 1
                )
            )
        elif key == "a":
            self.frame_indicator.setValue(max(0, self.current_frame - 1))
        elif key == " ":
            self.play_pause_clicked()

    def update_video(self):
        self.slider_widget.setMaximum(self.params["video_n_frames"]() - 1)
        self.frame_indicator.setMaximum(self.params["video_n_frames"]() - 1)
        self.im.set_extent(
            (
                0,
                *self.params["video_size"](),
                0,
            )
        )
        self.x_center = self.params["video_size"]()[0] / 2
        self.y_center = self.params["video_size"]()[1] / 2
        self.fit_zoom(*self.params["video_size"]())

        self.new_params()

    def update_mask(self, polygons):
        while self.mask_polygons:
            self.mask_polygons.pop().remove()

        for polygon in polygons:
            self.mask_polygons.append(self.ax.add_patch(polygon))
        self.new_params()

    def new_params(self):
        self.animal_detection_parameters = {
            key: value() for key, value in self.params.items()
        }

        # TODO is this necessary?
        if not self.animal_detection_parameters["ROI_mask"].any():
            self.animal_detection_parameters["ROI_mask"] = 0

        self.update_player()
