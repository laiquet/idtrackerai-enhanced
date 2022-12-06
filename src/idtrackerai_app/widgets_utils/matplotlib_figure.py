from math import sqrt

from matplotlib.backend_bases import MouseEvent, ResizeEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget


class MplCanvas(FigureCanvasQTAgg):
    click_event = pyqtSignal(int, float, float)
    new_drawn = pyqtSignal()

    def __init__(self, adapting_zoom=True):
        self.fig = Figure(facecolor="black")
        self.ax = self.fig.add_axes([0, 0, 1, 1], facecolor="black")
        super().__init__(self.fig)
        self.setFocusPolicy(Qt.TabFocus)

        self.zoom = 1
        self.adapting_zoom = adapting_zoom
        self.x_center = 0
        self.y_center = 0
        self.mouse_pressed = False
        self.has_moved = False

        self.ax.axis(False)

        self.canvas_size = self.fig.get_size_inches() * self.fig.dpi

        self.mpl_connect("button_press_event", self.on_click_press)
        self.mpl_connect("button_release_event", self.on_click_release)
        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.keyPressEvent = lambda event: event.ignore()
        self.keyReleaseEvent = lambda event: event.ignore()

    def on_click_press(self, event: MouseEvent):
        if event.dblclick:
            event.step = 3
            self.on_scroll(event)
            self.has_moved = True  # avoid click signal
        else:
            self.has_moved = False
            self.mouse_pressed = True
            self.click_origin = (event.x, event.y)

    def on_click_release(self, event: MouseEvent):
        self.mouse_pressed = False
        if not self.has_moved:
            self.click_event.emit(event.button, event.xdata, event.ydata)

    def on_scroll(self, event: MouseEvent):
        self.x_center += (event.xdata - self.x_center) * 0.1 * event.step
        self.y_center += (event.ydata - self.y_center) * 0.1 * event.step
        self.zoom *= 1 - 0.1 * event.step
        self.set_ax_lims()

    def on_motion(self, event: MouseEvent):
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

    def resizeEvent(self, event):
        """Overriding mpl.backends.backend_qt.FigureCanvasQT.resizeEvent()
        because there's a draw() in the original version that don't work for us.
        It would be great to find a better solution"""
        if self._in_resize_event:  # Prevent PyQt6 recursion
            return
        self._in_resize_event = True
        try:
            w = event.size().width() * self.device_pixel_ratio
            h = event.size().height() * self.device_pixel_ratio
            dpival = self.fig.dpi
            winch = w / dpival
            hinch = h / dpival
            self.fig.set_size_inches(winch, hinch, forward=False)
            # pass back into Qt to let it finish
            QWidget.resizeEvent(self, event)
            # emit our resize events
            ResizeEvent("resize_event", self)._process()
            old_diagonal = self.canvas_size[0] ** 2 + self.canvas_size[1] ** 2
            self.canvas_size = self.get_width_height()
            actual_diagonal = (
                self.canvas_size[0] ** 2 + self.canvas_size[1] ** 2
            )
            if self.adapting_zoom:
                self.zoom *= sqrt(old_diagonal / actual_diagonal)
            self.set_ax_lims()
        finally:
            self._in_resize_event = False

    def set_ax_lims(self):
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
        self.ax.set_visible(False)
        self.draw()
        self.ax.set_visible(True)
        self.bg = self.copy_from_bbox(self.fig.bbox)
        self.new_drawn.emit()
