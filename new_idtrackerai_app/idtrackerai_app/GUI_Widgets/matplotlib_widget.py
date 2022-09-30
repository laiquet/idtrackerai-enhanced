from matplotlib.pyplot import figure, rcParams


class matplotlib_gui:
    def draw_and_flush(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def __init__(self, remove_buildin_keybindings=True):

        self.zoom = 1

        self.x_center = 0
        self.y_center = 0
        self.mouse_pressed = False
        self.has_moved = False

        self.fig = figure(figsize=(1, 1))
        self.canvas = self.fig.canvas

        self.ax = self.fig.add_axes(
            [0, 0, 1, 1],
            xticks=(),
            yticks=(),
            facecolor="black",
        )

        self.canvas_size = self.fig.get_size_inches() * self.fig.dpi

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect(
            "button_release_event", self.on_click_release
        )
        # self.fig.canvas.mpl_connect("key_release_event", self.keyPressEvent)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

        if remove_buildin_keybindings:
            # Clean all the default keyboard shortcuts of matplotlib
            for action, keybindings in rcParams.items():
                if action.startswith("keymap."):
                    keybindings.clear()

    def on_click(self, event):
        self.has_moved = False
        self.mouse_pressed = True
        self.click_origin = (event.x, event.y)

    def on_click_release(self, event):
        self.mouse_pressed = False
        if not self.has_moved:
            if hasattr(self, f"click_in_plt_button_{event.button}"):
                getattr(self, f"click_in_plt_button_{event.button}")(event)

    # def on_key(self, *args):
    #     # print(event.key, "from matplotlib")
    #     self.keyPressEvent(*args)
    # try:
    #     int_key = int(event.key)
    # except ValueError:
    #     try:
    #         getattr(self, f"key_{event.key}")()
    #         print("key sended!")
    #     except AttributeError:
    #         print(f"no key {event.key}")
    #         pass
    # else:
    #     if hasattr(self, "key_number"):
    #         self.key_number(int_key)

    def on_scroll(self, event):
        self.x_center += (self.x_center - event.xdata) * 0.1 * event.step
        self.y_center += (self.y_center - event.ydata) * 0.1 * event.step
        self.zoom += 0.1 * self.zoom * event.step
        self.set_ax_lims()

    def on_motion(self, event):
        if self.mouse_pressed:
            self.has_moved = True
            self.x_center -= 2 * self.zoom * (event.x - self.click_origin[0])
            self.y_center += 2 * self.zoom * (event.y - self.click_origin[1])
            self.click_origin = (event.x, event.y)
            self.set_ax_lims()

    def on_resize(self, event):
        self.canvas_size = (event.width, event.height)
        self.set_ax_lims()

    def set_ax_lims(self, draw=True):
        self.ax.set(
            xlim=(
                self.x_center - self.zoom * self.canvas_size[0],
                self.x_center + self.zoom * self.canvas_size[0],
            ),
            ylim=(
                self.y_center + self.zoom * self.canvas_size[1],
                self.y_center - self.zoom * self.canvas_size[1],
            ),
        )
        if draw:
            self.fig.canvas.draw()
