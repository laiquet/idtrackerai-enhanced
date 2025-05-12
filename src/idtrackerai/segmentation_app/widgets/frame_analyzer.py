from collections.abc import Sequence

import numpy as np
from qtpy.QtCore import QPoint, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget

from idtrackerai.base.animals_detection import process_frame
from idtrackerai.GUI_tools import CanvasPainter


class FrameAnalyzer(QWidget):
    new_areas = Signal(int, list)
    new_parameters = Signal()

    def set_bkg(self, bkg_model):
        self.bkg_model = bkg_model
        self.use_bkg = bkg_model is not None
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_ROI_mask(self, ROI_mask: np.ndarray | None):
        self.ROI_mask = ROI_mask
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_intensity_ths(self, intensity_ths: Sequence[float]):
        self.intensity_ths = intensity_ths
        self.need_to_redraw = True
        self.new_parameters.emit()

    def set_area_ths(self, area_ths: Sequence[float]):
        self.area_ths = area_ths
        self.need_to_redraw = True
        self.new_parameters.emit()

    def __init__(self):
        super().__init__()

        self.use_bkg = False
        self.bkg_model = None
        self.ROI_mask = None
        self.intensity_ths = [0, 1]
        self.area_ths = [1, 1]
        self.blob_polygons: list[list[QPoint]] = []
        self.drawn_frame = -1

    def process_frame(self, frame: np.ndarray | None):
        if frame is None:  # error frame
            self.areas, contours = [], []
        else:
            self.areas, contours, gray_frame = process_frame(
                frame,
                bkg_model=self.bkg_model,
                ROI_mask=self.ROI_mask,
                intensity_ths=self.intensity_ths,
                area_ths=self.area_ths,
            )

        self.n_blobs = len(contours)
        for i, contour in enumerate(contours):
            polygon = [QPoint(*xy) for xy in contour]
            if i == len(self.blob_polygons):
                self.blob_polygons.append(polygon)
            else:
                self.blob_polygons[i] = polygon

    def paint_on_canvas(
        self, painter: CanvasPainter, frame_number: int, frame: np.ndarray | None
    ):
        if self.drawn_frame != frame_number or self.need_to_redraw:
            self.process_frame(frame)
            self.new_areas.emit(frame_number, self.areas)
            self.need_to_redraw = False
        painter.setBrush(QColor(60, 160, 255, 150))
        painter.setPen(QColor(0x286384))
        for i in range(self.n_blobs):
            painter.drawPolygon(self.blob_polygons[i])  # type: ignore
        self.drawn_frame = frame_number
