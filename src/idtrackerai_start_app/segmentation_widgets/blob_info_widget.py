from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QColorConstants, QPainter, QPaintEvent
from qtpy.QtWidgets import QWidget


class BlobInfoWidget(QWidget):
    bar_width = 0.65

    def __init__(self):
        super().__init__()
        self.areas = []
        self.frame = 0
        self.n_animals = 0
        self.tracking_intervals: list[list[int]] | None = None
        self.setMinimumSize(100, 100)

    def in_tracking_intervals(self, frame) -> bool:
        if self.tracking_intervals is None:
            return True
        return any(start <= frame < end for start, end in self.tracking_intervals)

    def setAreas(self, frame: int, areas: list[int]):
        self.frame = frame
        self.areas = areas
        self.update()

    def setNAnimals(self, n_animals):
        self.n_animals = n_animals
        self.update()

    def setTrackingIntervals(self, tracking_intervals):
        self.tracking_intervals = tracking_intervals
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        base_color = painter.pen().color()
        left_margin = 8
        w = self.width() - left_margin
        h = self.height()

        if w > 700:
            middle = int(0.5286 * w)
            left = middle - 300 + left_margin
            right = middle + 300
        else:
            left = 70 + left_margin
            right = w - 30
        axis_w = right - left

        if h > 500:
            middle = h // 2
            top = middle - 200
            bottom = middle + 220
        else:
            top = 50
            bottom = h - 30
        axis_h = bottom - top

        painter.drawText(
            0,
            bottom + 5,
            w,
            bottom + 50,
            Qt.AlignmentFlag.AlignHCenter,
            "Detected blobs",
        )
        painter.save()
        painter.translate(0, h)
        painter.rotate(-90)
        painter.drawText(
            0,
            0,
            h,
            left - 50,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            "Area (px)",
        )
        painter.restore()

        number_of_blobs = len(self.areas)

        scale = 1
        rects = []
        if not self.isEnabled():
            title = ""
            min_area_line = None
        elif not self.in_tracking_intervals(self.frame):
            title = "Frame outside tracking intervals"
            min_area_line = None
        elif number_of_blobs == 0:
            title = "No blobs detected"
            min_area_line = None
        else:
            if self.n_animals == 0:
                title_prefix = "# animals = 0! "
                painter.setBrush(QColor(0xBA2320))
                painter.setPen(QColor(0x5A1010))
            elif number_of_blobs > self.n_animals:
                title_prefix = "More blobs than animals! "
                painter.setBrush(QColor(0xBA2320))
                painter.setPen(QColor(0x5A1010))
            else:
                title_prefix = ""
                painter.setBrush(QColor(0x44A0D9))
                painter.setPen(QColor(0x286384))

            bar_sep = axis_w / number_of_blobs
            bar_width = 0.7 * axis_w / number_of_blobs
            scale = axis_h / (1.1 * max(self.areas))
            rects = [
                QRectF(
                    left + (i + 0.5) * bar_sep - 0.5 * bar_width,
                    bottom - area * scale,
                    bar_width,
                    area * scale,
                )
                for i, area in enumerate(self.areas)
            ]
            painter.drawRects(rects)  # type: ignore

            min_area_line = min(self.areas)
            if number_of_blobs == 1:
                title = f"1 blob detected of area {self.areas[0]:.0f} px"
            else:
                title = (
                    title_prefix
                    + f"{number_of_blobs} blobs detected. "
                    f"Minimum area: {min_area_line:.0f} px"
                )

        # Draw min area dashed line
        if min_area_line is not None:
            pen = painter.pen()
            pen.setColor(QColor(0x808080))
            pen.setStyle(Qt.PenStyle.DotLine)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(
                left,
                bottom - int(min_area_line * scale),
                right,
                bottom - int(min_area_line * scale),
            )

        # Draw title
        painter.setPen(
            QColorConstants.Red if title == "Number of animals missing!" else base_color
        )
        painter.drawText(
            0,
            0,
            w,
            top,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

        # Draw ticks
        painter.setPen(base_color)
        tick_lenght = 10
        if rects:
            lim = 1.1 * max(self.areas)
            for frequency in (1, 5, 10, 25, 50, 100, 200, 500, 100, 2000, 5000, 10000):
                n_ticks = int(lim / frequency)
                if n_ticks < 5:
                    break
            for tick_index in range(n_ticks + 1):
                tick_value = frequency * tick_index
                tick_high = int(bottom - tick_value * scale)
                painter.drawLine(left - tick_lenght, tick_high, left, tick_high)
                painter.drawText(
                    0,
                    tick_high - 20,
                    left - tick_lenght - 3,
                    40,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"{tick_value}",
                )

        # Draw axes
        painter.drawLine(left, bottom, right, bottom)
        painter.drawLine(left, bottom, left, top)
