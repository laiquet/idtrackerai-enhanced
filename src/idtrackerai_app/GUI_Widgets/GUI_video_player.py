from .matplotlib_widget import matplotlib_gui
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
from PyQt6.QtCore import Qt, QTimer
from functools import lru_cache
import cv2
from matplotlib.pyplot import subplots, rcParams
from confapp import conf
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
        self.canvas = self.fig.canvas
        self.min_area_line = self.ax.axhline(
            0, linestyle=":", color="gray", visible=False
        )
        self.bars = self.ax.bar([], [])

    def update(self, list_of_areas):
        number_of_blobs = len(list_of_areas)
        self.bars.remove()
        self.bars = self.ax.bar(
            range(number_of_blobs),
            list_of_areas,
            color="#44A0D9",
            edgecolor="#286384",
            width=0.65,
        )

        if number_of_blobs == 0:
            self.ax.set(title="No blobs detected")
            self.min_area_line.set_visible(False)
            self.ax.set(ylim=(0, 1))
        elif number_of_blobs == 1:
            self.ax.set(
                title=f"1 blob detected of area {list_of_areas[0]:.0f} px"
            )
            self.min_area_line.set_ydata(list_of_areas[0])
            self.min_area_line.set_visible(True)
            self.ax.set(ylim=(0, 1.1 * list_of_areas[0]), xlim=(-0.5, 0.5))
        elif number_of_blobs > 1:
            min_area = min(list_of_areas)
            self.ax.set(
                title=f"{number_of_blobs} blobs detected. Minimum area: {min_area:.0f} px"
            )
            self.min_area_line.set_ydata(min_area)
            self.min_area_line.set_visible(True)
            self.ax.set(
                ylim=(0, 1.1 * max(list_of_areas)),
                xlim=(-0.5, number_of_blobs - 0.5),
            )
        else:
            raise TypeError

        self.fig.canvas.draw()


class VideoPlayer(matplotlib_gui):
    def __init__(self, param_func, video_path=None, actual_conf=None):
        super().__init__()
        self.param_func = param_func
        self.canvas.setEnabled(False)
        self.video_holder = VideoHolder(video_path)
        self.params = {}

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

        self.blob_polygons = self.ax.fill()

        self.time_indicator_widget = QLabel()
        self.time_indicator_widget.setFixedHeight(24)

        self.play_pause_button = QPushButton(enabled=False)
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

        # self.setCentral
        # self.canvas.setFocusPolicy(Qt.StrongFocus)

        # self.zoom = 1
        # self.set_ax_lims()

        self.area_chart_widget = MplCanvas()
        self.area_chart_widget.canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.area_chart_widget.canvas.setVisible(False)
        self.VideoPlayer_layout = QVBoxLayout()
        self.VideoPlayer_layout.addWidget(self.area_chart_widget.canvas, 30)
        self.VideoPlayer_layout.addWidget(self.canvas, 62)
        self.VideoPlayer_layout.addLayout(self.control_bar, 8)

        self.current_frame = 0
        # self.update_player()
        self.time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_next_frame)
        self.mask_polygons = []

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

        # ret, thresh = cv2.threshold(frame, 145, 255, cv2.THRESH_BINARY)
        # out = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contours = out[0] if len(out) == 2 else out[1]

        # Save original shape to rescale if resolution reduction is applied
        # original_size = self.video_holder.size  # (width, height)
        # self._frame_width = original_size[0]
        # self._frame_height = original_size[1]
        # TODO: check if bkgmodel needs to be updated because of new ROI
        if isinstance(self.animal_detection_parameters["mask"], int):
            if self.animal_detection_parameters["mask"] == 0:
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

        # if animal_detection_parameters["resolution_reduction"] != 1:
        #     frame = cv2.resize(
        #         frame,
        #         None,
        #         fx=animal_detection_parameters["resolution_reduction"],
        #         fy=animal_detection_parameters["resolution_reduction"],
        #         interpolation=cv2.INTER_AREA,
        #     )
        # cv2.drawContours(frame, contours, -1, color=(0, 0, 255), thickness=-1)
        # Resize to original size (ROI and setup points are in original size)
        # frame = cv2.resize(frame, original_size, interpolation=cv2.INTER_AREA)
        # Draw ROIs in frame
        # self.draw_rois(frame)
        # Draw setup points in frame
        # self.draw_points_list(frame)
        # return frame

        for polygon in self.blob_polygons:
            polygon.remove()

        list_to_fill = []

        for contour in contours:
            list_to_fill.append(contour[..., 0])
            list_to_fill.append(contour[..., 1])
            # list_to_fill.append("#44A0D9")
        # print(list_to_fill[0].shape)
        self.blob_polygons = self.ax.fill(
            *list_to_fill, color="#44A0D9", edgecolor="#286384", lw=1
        )
        # color="#44A0D9",
        # edgecolor="#286384",

        self.area_chart_widget.update(areas)
        self.im.set_data(frame)
        self.draw_and_flush()

    def auto_next_frame(self):
        print(f" {1 / (perf_counter() - self.time):2.3f} fps", end="\r")
        self.time = perf_counter()
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
                self.video_holder.size[0],
                self.video_holder.size[1],
                0,
            )
        )
        self.x_center = self.video_holder.size[0] / 2
        self.y_center = self.video_holder.size[1] / 2
        self.set_ax_lims()

        self.current_frame = 0
        self.new_params()

    def update_mask(self, polygons):
        for patch in self.mask_polygons:
            patch.remove()
        self.mask_polygons = []

        for polygon in polygons:
            self.mask_polygons.append(self.ax.add_patch(polygon))
        self.new_params()

    def new_params(self):
        self.animal_detection_parameters = {
            "min_threshold": self.param_func["intensity_ths"]()[0],
            "max_threshold": self.param_func["intensity_ths"]()[1],
            "min_area": self.param_func["area_ths"]()[0],
            "max_area": self.param_func["area_ths"]()[1],
            "mask": self.param_func["ROI_mask"](),
            "subtract_bkg": self.param_func["bkg_check"](),
            "bkg_model": self.param_func["bkg"](),
            "resolution_reduction": self.param_func["resreduct"]() / 100,
            "sigma_gaussian_blurring": conf.SIGMA_GAUSSIAN_BLURRING,
        }

        mask = self.animal_detection_parameters["mask"]
        if not (mask).any():
            print("simplified mask to 0")
            self.animal_detection_parameters["mask"] = 0

        self.update_player()


class VideoHolder:
    """This class loads the `cv2.VideoCapture` object of the desired video path and provides the desired gray-scale frames with memoization in `frame(frame_number)`"""

    def __init__(self, path=None):
        if path:
            self.load(path)

    def load(self, path):
        self.path = path
        print(path)
        self.cap = cv2.VideoCapture(path)
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
