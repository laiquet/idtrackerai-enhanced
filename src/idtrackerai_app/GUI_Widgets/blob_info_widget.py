from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QVBoxLayout


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

        self.push_btn = QPushButton()
        self.push_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.push_btn.setText("▲")
        self.push_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.push_btn.clicked.connect(self.show_hide_event)
        self.push_btn.setFixedHeight(20)
        self.bars_visible = True
        self.areas = []
        self.frame = 0
        self.bg = None
        self.n_animals = 0
        self.tracking_intervals = [[0, 9999999999]]
        self.title = QLabel()
        self.title.setMaximumHeight(15)
        self.addWidget(self.title, alignment=Qt.AlignHCenter)
        self.addWidget(self.canvas, alignment=Qt.AlignHCenter)
        self.addWidget(self.push_btn)
        self.canvas.mpl_connect("draw_event", lambda x: self.draw(blit=False))
        self.bars = self.canvas.ax.bar([], [])

    def in_tracking_intervals(self, frame) -> bool:
        for start, end in self.tracking_intervals:
            if frame >= start and frame < end:
                return True
        return False

    def show_hide_event(self):
        self.bg = None
        self.bars_visible = not self.bars_visible
        self.title.setVisible(self.bars_visible)
        self.canvas.setVisible(self.bars_visible)
        if self.bars_visible:
            self.push_btn.setText("▲")
        else:
            self.push_btn.setText("▼")
        self.draw()

    def setAreas(self, frame: int, areas: list[int]):
        self.frame = frame
        self.areas = areas
        self.draw()

    def setNAnimals(self, n_animals):
        self.n_animals = n_animals
        self.draw()

    def setTrackingIntervals(self, tracking_intervals):
        self.tracking_intervals = tracking_intervals
        self.draw()

    def update_lims(self, ymax):
        self.canvas.ax.set(
            xlim=(
                -self.bar_width,
                max(1, len(self.areas)) - 1 + self.bar_width,
            )
        )

        actual_ylim = self.canvas.ax.get_ylim()[1]
        if ymax > actual_ylim or ymax < 0.7 * actual_ylim:
            self.canvas.ax.set(ylim=(0, 1.2 * ymax))
            self.bg = None

    def draw(self, blit=True):
        if not self.canvas.isVisible():
            return
        number_of_blobs = len(self.areas)

        if not self.in_tracking_intervals(self.frame):
            self.title.setText("Frame outside tracking intervals")
            self.min_area_line.set_visible(False)
            self.bars.remove()
            self.bars = self.canvas.ax.bar([], [], animate=True)

        else:

            if number_of_blobs > self.n_animals:
                title_prefix = "More blobs than animals! "

                facecolor = "#BA2320"
                edgecolor = "#5A1010"
            else:
                title_prefix = ""
                facecolor = "#44A0D9"
                edgecolor = "#286384"
            self.bars.remove()
            self.bars = self.canvas.ax.bar(
                range(number_of_blobs),
                self.areas,
                animated=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )

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
        for bar in self.bars:
            bar.draw(renderer)
        self.min_area_line.draw(renderer)
        if blit:
            self.canvas.blit()
