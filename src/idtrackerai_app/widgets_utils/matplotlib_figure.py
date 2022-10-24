from math import sqrt
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    click_on_plot = pyqtSignal(int, float, float)

    def draw_and_flush(self):
        self.draw()
        self.flush_events()

    def __init__(self, adapting_zoom=True):
        self.fig = Figure()
        self.ax = self.fig.add_axes(
            [0, 0, 1, 1],
            xticks=(),
            yticks=(),
            facecolor="black",
        )
        super().__init__(self.fig)
        self.setFocusPolicy(Qt.NoFocus)

        self.zoom = 1
        self.adapting_zoom = adapting_zoom
        self.x_center = 0
        self.y_center = 0
        self.mouse_pressed = False
        self.has_moved = False

        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        self.ax.spines.left.set_visible(False)
        self.ax.spines.bottom.set_visible(False)

        self.canvas_size = self.fig.get_size_inches() * self.fig.dpi

        self.mpl_connect("button_press_event", self.on_click)
        self.mpl_connect("button_release_event", self.on_click_release)
        # self.mpl_connect("key_release_event", self.keyPressEvent)
        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("resize_event", self.on_resize)

    def on_click(self, event):
        self.has_moved = False
        self.mouse_pressed = True
        self.click_origin = (event.x, event.y)

    def on_click_release(self, event):
        self.mouse_pressed = False
        if not self.has_moved:
            self.click_on_plot.emit(event.button, event.xdata, event.ydata)

    def on_scroll(self, event):
        self.x_center += (self.x_center - event.xdata) * 0.1 * event.step
        self.y_center += (self.y_center - event.ydata) * 0.1 * event.step
        self.zoom += 0.1 * self.zoom * event.step
        self.set_ax_lims()

    def on_motion(self, event):
        if self.mouse_pressed:
            self.has_moved = True
            self.x_center -= self.zoom * (event.x - self.click_origin[0])
            self.y_center += self.zoom * (event.y - self.click_origin[1])
            self.click_origin = (event.x, event.y)
            self.set_ax_lims()

    def fit_zoom(self, width, height, fit_to=None):
        if fit_to is None:
            fit_to = self.canvas_size
        canvas_ratio = fit_to[0] / fit_to[1]
        ratio_to_fit = width / height
        if canvas_ratio < ratio_to_fit:
            self.zoom = width / fit_to[0]
        else:
            self.zoom = height / fit_to[1]
        self.set_ax_lims()

    def on_resize(self, event):
        old_diagonal = self.canvas_size[0] ** 2 + self.canvas_size[1] ** 2
        actual_diagonal = event.width**2 + event.height**2
        if self.adapting_zoom:
            self.zoom *= sqrt(old_diagonal / actual_diagonal)
        self.canvas_size = (event.width, event.height)
        self.set_ax_lims()

    def set_ax_lims(self, draw=True):
        self.ax.set(
            xlim=(
                self.x_center - 0.5 * self.zoom * self.canvas_size[0],
                self.x_center + 0.5 * self.zoom * self.canvas_size[0],
            ),
            ylim=(
                self.y_center + 0.5 * self.zoom * self.canvas_size[1],
                self.y_center - 0.5 * self.zoom * self.canvas_size[1],
            ),
        )
        if draw:
            self.draw()
