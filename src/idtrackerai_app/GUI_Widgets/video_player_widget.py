from idtrackerai_app.widgets_utils import VideoPlayer
from matplotlib.patches import PathPatch, Polygon
from PyQt6.QtCore import pyqtSignal

from idtrackerai.animals_detection.segmentation import process_frame


class VideoPlayerWidget(VideoPlayer):
    new_areas = pyqtSignal(int, list)
    keys_for_segmentation = [
        "use_bkg",
        "bkg_model",
        "ROI_mask",
        "resolution_reduction",
        "intensity_ths",
        "area_ths",
    ]

    def __init__(self, parent, params):
        super().__init__()

        self.params = params

        self.blob_polygons = self.canvas.ax.fill()
        self.mask_polygons: list[PathPatch] = []
        self.frame_ready_to_draw.connect(self.process_frame)
        self.blobs_polys: list[Polygon] = []
        self.drawn_frame = -1

    def process_frame(self, renderer, new_params):
        current_frame = self.current_frame
        if not self.isEnabled():
            return
        if new_params:
            self.new_params()

        if new_params or (current_frame != self.drawn_frame):
            if isinstance(self.animal_detection_parameters["ROI_mask"], int):
                if self.animal_detection_parameters["ROI_mask"] == 0:
                    areas = []
                    contours = []
            else:
                areas, contours, gray_frame = process_frame(
                    self.VideoPathHolder.frame(current_frame),
                    **self.animal_detection_parameters,
                )

            resreduct = self.params["resolution_reduction"]()
            if resreduct != 1:
                contours = [contour / resreduct for contour in contours]

            for polygon in self.blob_polygons:
                polygon.remove()

            i = 0
            for i, contour in enumerate(contours):
                if i == len(self.blobs_polys):
                    self.blobs_polys.append(
                        self.canvas.ax.add_patch(
                            Polygon(
                                contour[:, 0, :],
                                closed=True,
                                facecolor="#44A0D9",
                                edgecolor="#286384",
                                lw=1,
                                animated=True,
                            )
                        )
                    )
                else:
                    self.blobs_polys[i].set_xy(contour[:, 0, :])
                self.blobs_polys[i].set_visible(True)
                self.blobs_polys[i].draw(renderer)
            for j in range(i + 1, len(self.blobs_polys)):
                self.blobs_polys[j].set_visible(False)

            self.new_areas.emit(current_frame, areas)
        else:
            for blob_polygon in self.blobs_polys:
                blob_polygon.draw(renderer)

    def new_params(self):
        self.animal_detection_parameters = {
            key: self.params[key]() for key in self.keys_for_segmentation
        }

        # When bkg is being computed, the bkg_model is None but use_bkg=True
        if self.animal_detection_parameters["bkg_model"] is None:
            self.animal_detection_parameters["use_bkg"] = False
        if not self.animal_detection_parameters["ROI_mask"].any():
            self.animal_detection_parameters["ROI_mask"] = 0
