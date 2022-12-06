from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from idtrackerai.animals_detection.segmentation import process_frame


class FrameAnalyzer(QWidget):
    new_areas = pyqtSignal(int, list)
    new_parameters = pyqtSignal()

    def set_bkg(self, bkg_model):
        self.bkg_model = bkg_model
        self.use_bkg = bkg_model is not None
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_ROI_mask(self, ROI_mask):
        self.ROI_mask = ROI_mask
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_resolution_reduction(self, resolution_reduction: float):
        self.resolution_reduction = resolution_reduction / 100
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_intensity_ths(self, intensity_ths: list[int]):
        self.intensity_ths = intensity_ths
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_area_ths(self, area_ths: list[int]):
        self.area_ths = area_ths
        self.need_to_redraw = True
        self.new_parameters.emit()

    def __init__(self, parent, ax: Axes):
        super().__init__()

        self.use_bkg = False
        self.bkg_model = None
        self.ROI_mask = None
        self.intensity_ths = [0, 1]
        self.area_ths = [0, 1]
        self.resolution_reduction = 1
        self.ax = ax
        self.blob_polygons = self.ax.fill()
        self.blobs_polys: list[Polygon] = []
        self.drawn_frame = -1

    def process_frame(self, frame):
        self.areas, contours, gray_frame = process_frame(
            frame,
            use_bkg=self.use_bkg,
            bkg_model=self.bkg_model,
            ROI_mask=self.ROI_mask,
            resolution_reduction=self.resolution_reduction,
            intensity_ths=self.intensity_ths,
            area_ths=self.area_ths,
        )

        if self.resolution_reduction != 1:
            contours = [
                contour / self.resolution_reduction for contour in contours
            ]

        for polygon in self.blob_polygons:
            polygon.remove()

        i = -1
        for i, contour in enumerate(contours):
            if i == len(self.blobs_polys):
                self.blobs_polys.append(
                    self.ax.add_patch(
                        Polygon(
                            contour[:, 0, :],  # type: ignore
                            closed=True,
                            facecolor="#44A0D9",
                            edgecolor="#286384",
                            lw=1,
                            animated=False,
                        )
                    )
                )
            else:
                self.blobs_polys[i].set_xy(contour[:, 0, :])
            self.blobs_polys[i].set_visible(True)
            # self.blobs_polys[i].(renderer)
        for j in range(i + 1, len(self.blobs_polys)):
            self.blobs_polys[j].set_visible(False)

    def draw_artists(self, renderer, frame_number, frame):
        if self.drawn_frame != frame_number or self.need_to_redraw:
            self.process_frame(frame)
            self.new_areas.emit(1, self.areas)  # TODO
            self.need_to_redraw = False
        for blob_polygon in self.blobs_polys:
            blob_polygon.draw(renderer)
        self.drawn_frame = frame_number
