import cv2
import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtGui import QColor, QPolygon
from qtpy.QtWidgets import QWidget

from idtrackerai.animals_detection.segmentation import process_frame
from idtrackerai_GUI_tools import CanvasPainter


class FrameAnalyzer(QWidget):
    new_areas = Signal(int, list)
    new_parameters = Signal()

    def set_bkg(self, bkg_model):
        self.bkg_model = bkg_model
        self.bkg_model_resreduct = bkg_model
        self.use_bkg = bkg_model is not None

        if bkg_model is not None and self.resolution_reduction != 1:
            self.bkg_model_resreduct = cv2.resize(
                self.bkg_model,
                None,  # type: ignore
                fx=self.resolution_reduction,
                fy=self.resolution_reduction,
                interpolation=cv2.INTER_AREA,
            )

        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_ROI_mask(self, ROI_mask: np.ndarray | None):
        self.ROI_mask = ROI_mask
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_resolution_reduction(self, resolution_reduction: float):
        self.resolution_reduction = resolution_reduction

        if resolution_reduction != 1:
            if self.bkg_model is not None:
                self.bkg_model_resreduct = cv2.resize(
                    self.bkg_model,
                    None,  # type: ignore
                    fx=resolution_reduction,
                    fy=resolution_reduction,
                    interpolation=cv2.INTER_AREA,
                )
        else:
            self.bkg_model_resreduct = self.bkg_model

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

    def __init__(self):
        super().__init__()

        self.use_bkg = False
        self.bkg_model = None
        self.bkg_model_resreduct = None
        self.ROI_mask = None
        self.intensity_ths = [0, 1]
        self.area_ths = [0, 1]
        self.resolution_reduction = 1
        self.blob_polygons: list[QPolygon] = []
        self.drawn_frame = -1

    def process_frame(self, frame):
        self.areas, contours, gray_frame = process_frame(
            frame,
            bkg_model=self.bkg_model_resreduct,
            ROI_mask=self.ROI_mask,
            resolution_reduction=self.resolution_reduction,
            intensity_ths=self.intensity_ths,
            area_ths=self.area_ths,
        )

        self.n_blobs = len(contours)
        for i, contour in enumerate(contours):
            if i == len(self.blob_polygons):
                self.blob_polygons.append(QPolygon())
            self.blob_polygons[i].setPoints(*contour.ravel())

    def paint_on_canvas(self, painter: CanvasPainter, frame_number: int, frame):
        if self.drawn_frame != frame_number or self.need_to_redraw:
            self.process_frame(frame)
            self.new_areas.emit(frame_number, self.areas)
            self.need_to_redraw = False
        painter.setBrush(QColor(60, 160, 255, 150))
        painter.setPenColor(QColor(0x286384))
        for i in range(self.n_blobs):
            painter.drawPolygon(self.blob_polygons[i])
        self.drawn_frame = frame_number
