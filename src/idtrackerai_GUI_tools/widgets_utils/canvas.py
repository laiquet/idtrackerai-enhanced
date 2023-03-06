import logging

from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPolygon,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget


class CustomPainter(QPainter):
    def __init__(self, parent, zoom: float):
        self.applied_zoom = zoom
        super().__init__(parent)

    def drawPolygonFromVertices(self, vertices):
        poly = QPolygon()
        poly.setPoints(*[coord for point in vertices for coord in point])
        super().drawPolygon(poly)

    def setPenColor(self, color: QColor | int):
        super().setPen(color)
        pen = self.pen()
        pen.setWidthF(1.3 * self.applied_zoom)
        super().setPen(pen)

    def drawBigPoint(self, x: float, y: float, size=7):
        radi = (size * self.applied_zoom) / 2
        super().drawEllipse(QPointF(x, y), radi, radi)


class Canvas(QWidget):
    # TODO check better implementations with
    # QGraphicsItem, QGraphicsScene, QtQuick, Canvas

    click_event = pyqtSignal(Qt.MouseButton, float, float, float)
    """button, current zoom, x data, y data"""
    double_click_event = pyqtSignal(Qt.MouseButton, float, float, float)
    """button, current zoom, x data, y data"""
    painting_time = pyqtSignal(CustomPainter)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.zoom = 3.0
        self.centerX: float = 0.0
        self.centerY: float = 0.0
        self.has_moved: bool = False
        self.mouse_pressed: bool = False
        self.click_origin: tuple = (0, 0)
        self.real_w_zoom: float
        self.real_h_zoom: float
        self.real_x0: int
        self.real_y0: int

    def paintEvent(self, event: QPaintEvent):
        painter = CustomPainter(self, self.zoom)
        try:
            painter_rect = event.rect()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(painter_rect, 0x000000)
            axis_w = painter_rect.width() * self.zoom
            axis_h = painter_rect.height() * self.zoom

            # save inaccuracies in rounding to use in self.to_physical_units
            self.real_w_zoom = int(axis_w) / painter_rect.width()
            self.real_h_zoom = int(axis_h) / painter_rect.height()
            self.real_x0 = int(self.centerX - axis_w / 2)
            self.real_y0 = int(self.centerY - axis_h / 2)

            painter.setWindow(self.real_x0, self.real_y0, int(axis_w), int(axis_h))

            font = self.font()
            font.setPointSizeF(font.pointSizeF() * 1.3 * self.zoom)
            painter.setFont(font)

            pen = painter.pen()
            pen.setWidthF(1.8 * self.zoom)
            painter.setPen(pen)

            self.painting_time.emit(painter)
        except Exception as e:
            logging.error(e)

    def to_physical_units(self, point: QPoint | QPointF):
        return (
            self.real_x0 + self.real_w_zoom * point.x(),
            self.real_y0 + self.real_h_zoom * point.y(),
        )

    def wheelEvent(self, event: QWheelEvent):
        step = event.angleDelta().y() / 1200
        if step > 0 and self.zoom < 0.1:
            return
        if step < 0 and self.zoom > 100:
            return
        xdata, ydata = self.to_physical_units(event.position())
        self.centerX += (xdata - self.centerX) * step
        self.centerY += (ydata - self.centerY) * step
        self.zoom *= 1 - step
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        self.has_moved = False
        self.mouse_pressed = True
        self.click_origin = (event.pos().x(), event.pos().y())

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # TODO emit a subclass of QMouseEvent with added attributes
        self.double_click_event.emit(
            event.button(), self.zoom, *self.to_physical_units(event.pos())
        )

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.mouse_pressed = False

        if not self.has_moved:
            self.setFocus()
            self.click_event.emit(
                event.button(), self.zoom, *self.to_physical_units(event.pos())
            )

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.mouse_pressed:
            self.has_moved = True

            self.centerX -= self.zoom * (event.pos().x() - self.click_origin[0])
            self.centerY -= self.zoom * (event.pos().y() - self.click_origin[1])
            self.click_origin = (event.pos().x(), event.pos().y())
            self.update()

    def adjust_zoom_to(self, width, height):
        self.centerX = width / 2
        self.centerY = height / 2
        self.zoom = max(width / self.width(), height / self.height())
