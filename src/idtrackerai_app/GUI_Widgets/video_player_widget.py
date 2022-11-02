from idtrackerai_app.widgets_utils import (
    MplCanvas,
    VideoPathHolder_Cls,
    VideoPlayer,
)
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
from idtrackerai.animals_detection.segmentation import _process_frame
import cv2
from functools import lru_cache


class VideoPlayerWidget(VideoPlayer):
    new_areas = pyqtSignal(int, list)

    def __init__(self, parent, params):
        super().__init__()

        self.params = params

        self.blob_polygons = self.canvas.ax.fill()
        self.mask_polygons = []
        self.frame_ready_to_draw.connect(self.process_frame)

    def process_frame(self):
        current_frame = self.current_frame
        if not self.isEnabled():
            return
        if isinstance(self.animal_detection_parameters["ROI_mask"], int):
            if self.animal_detection_parameters["ROI_mask"] == 0:
                areas = []
                contours = []
        else:
            (_, _, _, areas, _, contours, _,) = _process_frame(
                self.VideoPathHolder.frame(current_frame),
                current_frame,
                save_pixels="NONE",
                save_segmentation_image="NONE",
                **self.animal_detection_parameters,
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
        self.blob_polygons = self.canvas.ax.fill(
            *list_to_fill,
            color="#44A0D9",
            edgecolor="#286384",
            lw=1,
        )

        self.new_areas.emit(current_frame, areas)

    def update_mask(self, ROI_patches):

        while self.mask_polygons:
            self.mask_polygons.pop().remove()

        for polygon in ROI_patches:
            self.mask_polygons.append(self.canvas.ax.add_patch(polygon))
        self.new_params()

    def new_params(self):

        keys_for_segmentation = [
            "use_bkg",
            "bkg_model",
            "ROI_mask",
            "resolution_reduction",
            "intensity_ths",
            "area_ths",
        ]

        self.animal_detection_parameters = {
            key: self.params[key]() for key in keys_for_segmentation
        }

        # When bkg is being computed, the bkg_model is None but use_bkg=True
        if self.animal_detection_parameters["bkg_model"] is None:
            self.animal_detection_parameters["use_bkg"] = False
        if not self.animal_detection_parameters["ROI_mask"].any():
            self.animal_detection_parameters["ROI_mask"] = 0

        self.update_player()
