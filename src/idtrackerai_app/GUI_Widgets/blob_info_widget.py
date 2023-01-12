from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class BlobInfoWidget(QWidget):
    bar_width = 0.65

    def __init__(self, parent):
        super().__init__()

        self.bars_visible = True
        self.areas = []
        self.frame = 0
        self.bg = None
        self.n_animals = 0
        self.tracking_intervals = [[0, 9999999999]]
        self.setMinimumSize(100, 100)
        # self.title = QLabel()
        # self.title.setMaximumHeight(15)
        # self.setLayout(QVBoxLayout())
        # self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        # self.layout().setAlignment(Qt.AlignmentFlag.AlignHCenter)
        # self.layout().addWidget(self.title)
        # self.layout().addWidget(self.canvas)
        # self.canvas.mpl_connect("draw_event", lambda x: self.draw(blit=False))
        # self.bars = self.canvas.ax.bar([], [])

    def in_tracking_intervals(self, frame) -> bool:
        for start, end in self.tracking_intervals:
            if frame >= start and frame < end:
                return True
        return False

    def show_hide_event(self):
        self.bg = None
        self.bars_visible = not self.bars_visible
        # self.title.setVisible(self.bars_visible)
        # self.canvas.setVisible(self.bars_visible)
        self.update()

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
        w = self.width()
        h = self.height()

        if w > 500:
            middle = w // 2
            left = middle - 200
            right = middle + 200
        else:
            left = 50
            right = w - 50
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
            left - 30,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            "Area in pixels",
        )
        painter.restore()

        number_of_blobs = len(self.areas)

        scale = 1

        if not self.in_tracking_intervals(self.frame):
            title = "Frame outside tracking intervals"
            rects = []
            min_area_line = None
        else:
            if number_of_blobs > self.n_animals:
                title_prefix = "More blobs than animals! "
                facecolor = "#BA2320"
                edgecolor = "#5A1010"
            else:
                title_prefix = ""
                facecolor = "#44A0D9"
                edgecolor = "#286384"
            bar_sep = axis_w / number_of_blobs
            bar_width = 0.7 * axis_w / number_of_blobs
            scale = axis_h / max(self.areas)
            rects = [
                (
                    int(left + (i + 0.5) * bar_sep - 0.5 * bar_width),
                    int(bottom - area * scale),
                    int(bar_width),
                    int(area * scale),
                )
                for i, area in enumerate(self.areas)
            ]

            if number_of_blobs == 0:
                title = "No blobs detected"
                min_area_line = None
            elif number_of_blobs == 1:
                title = f"1 blob detected of area {self.areas[0]:.0f} px"
                min_area_line = self.areas[0]
            elif number_of_blobs > 1:
                min_area_line = min(self.areas)
                title = (
                    f"{number_of_blobs} blobs detected. {title_prefix}"
                    f"Minimum area: {min_area_line:.0f} px"
                )

        painter.drawText(
            0,
            0,
            w,
            top,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

        painter.setBrush(QColor(facecolor))
        painter.setPen(QColor(edgecolor))
        for rect in rects:
            painter.drawRect(*rect)
        pen = painter.pen()
        if min_area_line is not None:

            pen.setColor(QColor(128, 128, 128))
            pen.setStyle(Qt.PenStyle.DotLine)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(
                left,
                bottom - int(min_area_line * scale),
                right,
                bottom - int(min_area_line * scale),
            )
        painter.setPen(base_color)
        painter.drawLine(left, bottom, right, bottom)
        painter.drawLine(left, bottom, left, top)
