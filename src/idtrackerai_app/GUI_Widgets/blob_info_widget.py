from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCommonStyle,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
)


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
    bar_width = 0.65

    def __init__(self, parent):
        super().__init__()
        self.canvas = MplStaticCanvas(parent)

        self.canvas.ax.set(
            xticks=(), ylabel="Area in pixels", xlabel="Detected blobs"
        )
        self.min_area_line = self.canvas.ax.axhline(
            0, linestyle=":", color="gray", animated=True
        )
        self.bars: list[Rectangle] = []

        self.hide_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarShadeButton
        )
        self.show_icon = QCommonStyle().standardIcon(
            QStyle.StandardPixmap.SP_TitleBarUnshadeButton
        )

        self.push_btn = QPushButton()
        self.push_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.push_btn.setIcon(self.hide_icon)
        self.push_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.push_btn.clicked.connect(self.show_hide_event)
        self.push_btn.setFixedHeight(15)
        self.bars_visible = True
        self.areas = []
        self.frame = 0
        self.bg = None
        self.n_animals = 0
        self.tracking_intervals = [[0, 9999999999]]
        self.title = QLabel()
        self.addWidget(self.title, alignment=Qt.AlignCenter)
        self.addWidget(self.canvas, alignment=Qt.AlignCenter)
        self.addWidget(self.push_btn)
        self.canvas.mpl_connect("draw_event", lambda x: self.draw(blit=False))

    def in_tracking_intervals(self, frame) -> bool:
        for start, end in self.tracking_intervals:
            if frame >= start and frame < end:
                return True
        return False

    def show_hide_event(self):
        self.bars_visible = not self.bars_visible
        self.canvas.setVisible(self.bars_visible)
        self.title.setVisible(self.bars_visible)
        self.bg = None
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

    def update_bars(self, heights):
        # TODO simplify with ax.bars (check performance)

        of = self.bar_width / 2
        n_bars = len(heights)
        current_bars = len(self.bars)

        i = -1
        for i in range(min(current_bars, n_bars)):
            self.bars[i].set_height(heights[i])
            self.bars[i].set_visible(True)
        for j in range(i + 1, n_bars):
            self.bars.append(
                self.canvas.ax.add_patch(
                    Rectangle(
                        (j - of, 0),
                        self.bar_width,
                        heights[j],
                        animated=True,
                    )
                )
            )
        for j in range(i + 1, current_bars):
            self.bars[j].set_visible(False)

    def update_lims(self, ymax):
        self.canvas.ax.set(
            xlim=(
                -self.bar_width,
                max(1, len(self.areas)) - 1 + self.bar_width,
            )
        )

        actual_ylim = self.canvas.ax.get_ylim()[1]
        if ymax > actual_ylim or ymax < 0.8 * actual_ylim:
            self.canvas.ax.set(ylim=(0, 1.1 * ymax))
            self.bg = None

    def draw(self, blit=True):
        if not self.canvas.isVisible():
            return
        number_of_blobs = len(self.areas)

        if not self.in_tracking_intervals(self.frame):
            self.title.setText("Frame outside tracking intervals")
            self.min_area_line.set_visible(False)
            self.update_bars([])
        else:

            title_prefix = (
                "More blobs than animals! "
                if number_of_blobs > self.n_animals
                else ""
            )

            self.update_bars(self.areas)

            if number_of_blobs == 0:
                self.title.setText("No blobs detected")
                self.min_area_line.set_visible(False)
            elif number_of_blobs == 1:
                self.title.setText(
                    f"1 blob detected of area {self.areas[0]:.0f} px"
                )
                self.min_area_line.set_ydata(self.areas[0])
                self.min_area_line.set_visible(True)
                self.update_lims(self.areas[0])
            elif number_of_blobs > 1:
                min_area = min(self.areas)
                self.title.setText(
                    f"{number_of_blobs} blobs detected. {title_prefix}"
                    f"Minimum area: {min_area:.0f} px"
                )
                self.min_area_line.set_ydata(min_area)
                self.min_area_line.set_visible(True)
                self.update_lims(ymax=max(self.areas))

        if blit:
            if self.bg is None:
                self.canvas.setVisible(False)  # avoid inifite loops
                self.canvas.draw()
                self.canvas.setVisible(True)
                self.bg = self.canvas.copy_from_bbox(self.canvas.fig.bbox)
            else:
                self.canvas.restore_region(self.bg)

        renderer = self.canvas.get_renderer()
        if number_of_blobs > self.n_animals:
            for bar in self.bars:
                bar.set(facecolor="#BA2320", edgecolor="#5A1010")
                bar.draw(renderer)
        else:
            for bar in self.bars:
                bar.set(facecolor="#44A0D9", edgecolor="#286384")
                bar.draw(renderer)
        self.min_area_line.draw(renderer)
        if blit:
            self.canvas.blit()
