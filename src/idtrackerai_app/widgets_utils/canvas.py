import sys
from typing import overload

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QMoveEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QPolygon,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CustomQPainter(QPainter):
    def __init__(self, parent, zoom: float):
        self.applied_zoom = zoom
        super().__init__(parent)

    def drawPolygonFromVertices(self, vertices):
        poly = QPolygon()
        poly.setPoints(*[coord for point in vertices for coord in point])
        super().drawPolygon(poly)

    def setPenColor(self, color: QColor):
        super().setPen(color)
        pen = self.pen()
        pen.setWidthF(1.8 * self.applied_zoom)
        super().setPen(pen)

    def drawBigPoint(self, x: float, y: float, size=7):
        size *= self.applied_zoom
        size2 = size / 2
        super().drawEllipse(QRectF(x - size2, y - size2, size, size))


class Canvas(QWidget):
    click_event = pyqtSignal(int, float, float)
    painting_time = pyqtSignal(QPainter)

    def __init__(self, parent):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        # self.setStyleSheet("background-color:black;")

        # pal = QPalette()
        # pal.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        # self.setAutoFillBackground(True)
        # self.setPalette(pal)

        self.image_2_display = []
        self.img = None
        self.zoom = 3
        self.centerX = 0
        self.centerY = 0
        self.diagonal = None

    def resizeEvent(self, event: QResizeEvent):
        if self.diagonal is None:
            return
        actual_diagonal = self.height() ** 2 + self.width() ** 2
        self.zoom *= np.sqrt(self.diagonal / actual_diagonal)
        self.diagonal = actual_diagonal

    def paintEvent(self, event: QPaintEvent):
        painter = CustomQPainter(self, self.zoom)
        try:
            painter.fillRect(self.rect(), QColor("black"))
            axis_w = int(self.width() * self.zoom)
            axis_h = int(self.height() * self.zoom)

            painter.setWindow(
                int(self.centerX - axis_w / 2),
                int(self.centerY - axis_h / 2),
                axis_w,
                axis_h,
            )

            font = self.font()
            font.setPointSizeF(font.pointSizeF() * 1.3 * self.zoom)
            painter.setFont(font)
            # painter.drawImage(0, 0, self.img)

            pen = painter.pen()
            pen.setWidthF(1.8 * self.zoom)
            painter.setPen(pen)

            self.painting_time.emit(painter)

            # polygon = QPolygon()
            # arr = np.array([[10, 10], [100, 10], [100, 100], [10, 100]])
            # polygon.setPoints(*arr.ravel())
            # painter.drawPolygon(polygon, fillRule=Qt.FillRule.WindingFill)
            # painter.setPen(2)
            # painter.drawText(0, 0, "Qt")
        except Exception as e:
            print(e)
        # painter.end()

    def to_physical_units(self, point: QPoint | QPointF):
        return (
            self.centerX + self.zoom * (point.x() - self.width() / 2) - 0.5,
            self.centerY + self.zoom * (point.y() - self.height() / 2) - 0.5,
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

    def paint(self, frame: np.ndarray):
        self.centerX = frame.shape[1] // 2
        self.centerY = frame.shape[0] // 2
        self.img = QImage(
            frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888
        )
        self.update()

    def mousePressEvent(self, event: QMouseEvent):

        # if event.dblclick:
        #     event.step = 3
        #     self.on_scroll(event)
        #     self.has_moved = True  # avoid click signal
        # else:
        self.has_moved = False
        self.mouse_pressed = True
        self.click_origin = (event.pos().x(), event.pos().y())

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.mouse_pressed = False

        if not self.has_moved:
            self.click_event.emit(event.button(), *self.to_physical_units(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.mouse_pressed:
            self.has_moved = True

            self.centerX -= self.zoom * (event.pos().x() - self.click_origin[0])
            self.centerY -= self.zoom * (event.pos().y() - self.click_origin[1])
            self.click_origin = (event.pos().x(), event.pos().y())
            self.update()

    # def keyPressEvent(self, event: QKeyEvent):
    #     print(event)

    # def keyReleaseEvent(self, event: QKeyEvent):
    #     print(event)
