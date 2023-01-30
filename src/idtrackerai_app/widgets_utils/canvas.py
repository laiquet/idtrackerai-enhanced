from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPolygon,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget


class CustomQPainter(QPainter):
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
        size *= self.applied_zoom
        size2 = size / 2
        super().drawEllipse(QRectF(x - size2, y - size2, size, size))


class Canvas(QWidget):
    click_event = pyqtSignal(int, float, float)
    double_click_event = pyqtSignal(int, float, float)
    painting_time = pyqtSignal(QPainter)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.image_2_display = []
        self.img = None
        self.zoom = 3.0
        self.centerX = 0
        self.centerY = 0
        self.has_moved: bool = False

    def paintEvent(self, event: QPaintEvent):
        painter = CustomQPainter(self, self.zoom)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), 0x000000)
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

            pen = painter.pen()
            pen.setWidthF(1.8 * self.zoom)
            painter.setPen(pen)

            self.painting_time.emit(painter)
        except Exception as e:
            print(e)

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

    def mousePressEvent(self, event: QMouseEvent):
        self.has_moved = False
        self.mouse_pressed = True
        self.click_origin = (event.pos().x(), event.pos().y())

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.double_click_event.emit(
            event.button(), *self.to_physical_units(event.pos())
        )

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.mouse_pressed = False

        if not self.has_moved:
            self.setFocus()
            self.click_event.emit(event.button(), *self.to_physical_units(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.mouse_pressed:
            self.has_moved = True

            self.centerX -= self.zoom * (event.pos().x() - self.click_origin[0])
            self.centerY -= self.zoom * (event.pos().y() - self.click_origin[1])
            self.click_origin = (event.pos().x(), event.pos().y())
            self.update()

    def adjust_zoom_to(self, width, height):
        self.centerX = width // 2
        self.centerY = height // 2
        self.zoom = max(width / self.width(), height / self.height())
