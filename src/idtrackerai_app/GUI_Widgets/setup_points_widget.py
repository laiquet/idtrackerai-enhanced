from PyQt6.QtWidgets import QInputDialog
from idtrackerai_app.widgets_utils import ListLayout
import re
from PyQt6.QtCore import Qt
import ast


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = re.compile(r"[@!$%^&*?/\~:|]")
    return regex.search(string) is not None


# TODO fix reading/writing data
class SetupPointsWidget(ListLayout):
    def __init__(self, parent):
        super().__init__(name="Setup Points", parent=parent)
        self.add.clicked.connect(self.add_clicked)
        self.setup_points_dict = {}

        self.CheckBox.clicked.connect(self.CheckBox_changed_visible)
        self.ListChanged.connect(self.update_legend)

    def CheckBox_changed_visible(self, enabled):
        for i in range(self.list.count()):
            name = self.list.item(i).data(Qt.UserRole).split(":")[0]
            self.setup_points_dict[name].set(visible=enabled)
        self.draw_and_flush.emit()

    def add_clicked(self, checked):
        if checked:
            existing_names = [
                self.list.item(i).text().split(":")[0]
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

        else:
            points = self.plot_line.get_xydata().astype(int)
            self.plot_line.set_data([], [])

            self.setup_points_dict[self.setup_name] = self.ax.plot(
                *points.T, ".", label=self.setup_name
            )[0]

            self.add_str_to_list(
                self.setup_name
                + ": "
                + ",".join([f"{x,y}" for x, y in points])
            )

    def update_legend(self):
        legend_needed = self.CheckBox.isChecked() and self.list.count()
        if legend_needed:
            self.ax.legend()
        else:
            self.ax.legend([]).set_visible(False)

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.setup_points_dict.pop(
            item.data(Qt.UserRole).split(":")[0]
        ).remove()
        self.list.takeItem(self.list.row(item))

    def readList(self):
        out = {}
        for i in range(self.list.count()):
            text = self.list.item(i).data(Qt.UserRole)
            name, points = text.split(":")
            out[name] = ast.literal_eval(points)
        return out

    def setValue(self, values):
        if not values:
            return
        if isinstance(values, str):
            values = [values]

        self.list.clear()
        while self.setup_points_dict:
            self.setup_points_dict.pop().remove()

        self.CheckBox.click()

        for value in values:
            name, points_str = value.split(":")
            points = ast.literal_eval(points_str)
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            self.setup_points_dict[name] = self.ax.plot(x, y, ".", label=name)[
                0
            ]
            self.add_str_to_list(value)
