from PyQt6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QStyle,
    QCommonStyle,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplStaticCanvas(FigureCanvasQTAgg):
    def setColor(self, color):
        self.ax.tick_params(which="both", colors=color)
        self.ax.spines["bottom"].set_color(color)
        self.ax.spines["left"].set_color(color)
        self.title.set_color(color)
        self.ax.yaxis.label.set_color(color)
        self.ax.xaxis.label.set_color(color)

    def setEnabled(self, enabled):
        if enabled:
            self.setColor(self.enabled_color)
        else:
            self.setColor(self.disabled_color)
        super().setEnabled(enabled)

    def __init__(self, parent):
        self.fig = Figure(
            facecolor=parent.palette().window().color().name(),
            constrained_layout=True,
        )
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(facecolor="None")
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        self.enabled_color = parent.palette().windowText().color().name()
        self.disabled_color = (
            parent.palette().windowText().color().darker().name()
        )
        self.title = self.fig.suptitle("")


class BlobInfoWidget(QVBoxLayout):
    def __init__(self, parent):
        super().__init__()
        self.canvas = MplStaticCanvas(parent)

        self.canvas.ax.set(
            xticks=(), ylabel="Area in pixels", xlabel="Detected blobs"
        )
        self.min_area_line = self.canvas.ax.axhline(
            0, linestyle=":", color="gray", visible=False
        )
        self.bars = self.canvas.ax.bar([], [])

        self.hide_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarShadeButton
        )
        self.show_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarUnshadeButton
        )

        self.push_btn = QPushButton()
        self.push_btn.setIcon(self.hide_icon)
        self.push_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.push_btn.clicked.connect(self.show_hide_event)
        self.push_btn.setFixedHeight(15)
        self.bars_visible = True
        self.areas = []
        self.frame = 0
        self.n_animals = 0
        self.tracking_intervals = [[0, 9999999999]]
        self.addWidget(self.canvas, alignment=Qt.AlignCenter)
        self.addWidget(self.push_btn)

    def in_tracking_intervals(self, frame):
        for start, end in self.tracking_intervals:
            if frame >= start and frame < end:
                return True
        return False

    def show_hide_event(self):
        self.bars_visible = not self.bars_visible
        self.canvas.setVisible(self.bars_visible)
        if self.bars_visible:
            self.push_btn.setIcon(self.hide_icon)
        else:
            self.push_btn.setIcon(self.show_icon)

    def setAreas(self, frame, areas):
        self.frame = frame
        self.areas = areas
        self.draw()

    def setNAnimals(self, n_animals):
        self.n_animals = n_animals
        self.draw()

    def setTrackingIntervals(self, tracking_intervals):
        self.tracking_intervals = tracking_intervals
        self.draw()

    def draw(self):
        number_of_blobs = len(self.areas)

        self.bars.remove()

        if not self.in_tracking_intervals(self.frame):
            self.canvas.title.set_text("Frame outside tracking intervals")
            self.min_area_line.set_visible(False)
            self.canvas.ax.set(ylim=(0, 1))
            self.bars = self.canvas.ax.bar(
                [],
                [],
            )
        else:

            if number_of_blobs > self.n_animals:
                color = "#BA2320"
                edgecolor = "#5A1010"
                title_prefix = "More blobs than animals! "
            else:
                color = "#44A0D9"
                edgecolor = "#286384"
                title_prefix = ""

            self.bars = self.canvas.ax.bar(
                range(number_of_blobs),
                self.areas,
                color=color,
                edgecolor=edgecolor,
                width=0.65,
            )

            if number_of_blobs == 0:
                self.canvas.title.set_text("No blobs detected")
                self.min_area_line.set_visible(False)
                self.canvas.ax.set(ylim=(0, 1))
            elif number_of_blobs == 1:
                self.canvas.title.set_text(
                    f"1 blob detected of area {self.areas[0]:.0f} px"
                )
                self.min_area_line.set_ydata(self.areas[0])
                self.min_area_line.set_visible(True)
                self.canvas.ax.set(
                    ylim=(0, 1.1 * self.areas[0]), xlim=(-0.5, 0.5)
                )
            elif number_of_blobs > 1:
                min_area = min(self.areas)
                self.canvas.title.set_text(
                    f"{number_of_blobs} blobs detected. {title_prefix}"
                    f"Minimum area: {min_area:.0f} px"
                )
                self.min_area_line.set_ydata(min_area)
                self.min_area_line.set_visible(True)
                self.canvas.ax.set(
                    ylim=(0, 1.1 * max(self.areas)),
                    xlim=(-0.5, number_of_blobs - 0.5),
                )

        self.canvas.draw()
