from ast import literal_eval
from re import compile


from idtrackerai_app.widgets_utils import ListLayout, CustomQPainter
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QInputDialog


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = compile(r"[@!$%^&*?/\~:|]")
    return regex.search(string) is not None


class SetupPointsWidget(ListLayout):
    def __init__(self, parent):
        super().__init__(name="Setup Points", parent=parent)
        self.add.clicked.connect(self.add_clicked)
        self.setup_points_dict: dict[str, list[tuple[float, float]]] = {}
        self.ListChanged.connect(self.update_legend)
        self.ListChanged.connect(self.needToDraw.emit)
        self.update_legend()

    def click_event(self, button, x, y):
        if self.setup_name is not None:
            self.setup_points_dict[self.setup_name].append((x, y))
            self.needToDraw.emit()

    def add_clicked(self, checked):
        if checked:
            existing_names = [
                self.list.item(i).data(Qt.ItemDataRole.UserRole).split(":")[0]
                for i in range(self.list.count())
            ]
            invalid = True
            dialog_text = "Enter setup points name:"
            while invalid:
                name, ok = QInputDialog.getText(self.add, "idtracker.ai", dialog_text)
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

            self.setup_points_dict[name] = []
            self.update_legend()
            self.needToDraw.emit()

        else:
            assert self.setup_name is not None
            self.add_str_to_list(
                self.setup_name
                + ": "
                + ",".join(
                    [
                        f"[{x:.1f}, {y:.1f}]"
                        for x, y in self.setup_points_dict[self.setup_name]
                    ]
                )
            )
            self.setup_name = None

    def update_legend(self):
        return
        self.legend = self.ax.legend(handles=self.setup_points_dict.values(), loc=1)
        self.legend.set(animated=True)

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.setup_points_dict.pop(item.data(Qt.ItemDataRole.UserRole).split(":")[0])
        self.list.takeItem(self.list.row(item))
        self.update_legend()
        self.needToDraw.emit()

    def setValue(self, values):
        if not values:
            return
        if isinstance(values, str):
            values = [values]

        self.list.clear()
        self.setup_points_dict.clear()

        self.CheckBox.setChecked(True)

        for value in values:
            name, points_str = value.split(":")
            self.setup_points_dict[name] = literal_eval(points_str)
            self.add_str_to_list(value)

    def draw_artists(self, painter: CustomQPainter):
        if self.CheckBox.isChecked():

            # self.legend.draw(renderer)
            for points in self.setup_points_dict.values():
                for point in points:
                    painter.drawBigPoint(*point)
