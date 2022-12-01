from ast import literal_eval
from re import compile

import numpy as np
from idtrackerai_app.widgets_utils import ListLayout
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QInputDialog


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = compile(r"[@!$%^&*?/\~:|]")
    return regex.search(string) is not None


class SetupPointsWidget(ListLayout):
    def __init__(self, parent, ax: Axes):
        super().__init__(name="Setup Points", parent=parent)
        self.add.clicked.connect(self.add_clicked)
        self.setup_points_dict: dict[str, Line2D] = {}
        self.CheckBox.clicked.connect(self.uncheck_add)
        self.ax = ax
        self.ListChanged.connect(self.update_legend)
        self.update_legend()

    def click_event(self, button, x, y):
        if self.add.isChecked():
            artist = self.setup_points_dict[self.setup_name]
            xy = artist.get_xydata()
            artist.set_data(np.vstack([xy, (x, y)]).T)
            self.update_player.emit(False)

    def uncheck_add(self, enabled):
        if self.add.isChecked():
            self.add.click()

    def add_clicked(self, checked):
        if checked:
            existing_names = [
                self.list.item(i).data(Qt.UserRole).split(":")[0]
                for i in range(self.list.count())
            ]
            invalid = True
            dialog_text = "Enter setup points name:"
            while invalid:
                name, ok = QInputDialog.getText(
                    self.add, "idtracker.ai", dialog_text
                )
                if not ok:
                    self.add.setChecked(False)
                    break
                name = name.strip()
                invalid = has_invalid_chars(name)
                if invalid:
                    dialog_text = "Invalid characters encountered"
                if name in existing_names:
                    invalid = True
                    dialog_text = "Enter a non existing name"

            self.setup_name = name

            self.setup_points_dict[name] = self.ax.plot(
                [], [], ".", label=self.setup_name, animated=True
            )[0]
            self.update_legend()
            self.update_player.emit(False)

        else:

            self.add_str_to_list(
                self.setup_name
                + ": "
                + ",".join(
                    [
                        f"[{x:.1f}, {y:.1f}]"
                        for x, y in self.setup_points_dict[
                            self.setup_name
                        ].get_xydata()
                    ]
                )
            )
            self.setup_name = None

    def update_legend(self):
        self.legend = self.ax.legend(
            handles=self.setup_points_dict.values(), loc=1
        )
        self.legend.set(animated=True)

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.setup_points_dict.pop(
            item.data(Qt.UserRole).split(":")[0]
        ).remove()
        self.list.takeItem(self.list.row(item))
        self.update_legend()
        self.update_player.emit(False)

    def setValue(self, values):
        if not values:
            return
        if isinstance(values, str):
            values = [values]

        self.list.clear()
        while self.setup_points_dict:
            self.setup_points_dict.pop().remove()

        self.CheckBox.setChecked(True)

        for value in values:
            name, points_str = value.split(":")
            points = literal_eval(points_str)
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            self.setup_points_dict[name] = self.ax.plot(x, y, ".", label=name)[
                0
            ]
            self.add_str_to_list(value)

    def draw_artists(self, renderer):
        if self.CheckBox.isChecked():
            self.legend.draw(renderer)
            for artist in self.setup_points_dict.values():
                artist.draw(renderer)
