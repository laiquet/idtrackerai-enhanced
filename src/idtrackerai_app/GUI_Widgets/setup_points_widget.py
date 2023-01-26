from ast import literal_eval
from re import compile

from idtrackerai_app.widgets_utils import CustomQPainter, ListLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QInputDialog


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = compile(r"[@!$%^&*?/\~:|]")
    return regex.search(string) is not None


QColors = [
    QColor("#9467bd"),
    QColor("#2ca02c"),
    QColor("#bcbd22"),
    QColor("#ff7f0e"),
    QColor("#8c564b"),
    QColor("#e377c2"),
    QColor("#7f7f7f"),
    QColor("#17becf"),
]
n_colors = len(QColors)


class SetupPointsWidget(ListLayout):
    def __init__(self, parent):
        super().__init__(name="Setup Points")
        self.add.clicked.connect(self.add_clicked)
        self.setup_points_dict: dict[str, tuple[QColor, list[tuple[float, float]]]] = {}
        self.ListChanged.connect(self.needToDraw.emit)
        self.color_count = -1
        self.setup_name = None

    def click_event(self, button, x, y):
        if self.setup_name is not None:
            self.setup_points_dict[self.setup_name][1].append((x, y))
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
            self.color_count = (
                0 if self.color_count == n_colors - 1 else self.color_count + 1
            )

            self.setup_name = name
            self.setup_points_dict[name] = (QColors[self.color_count], [])
            self.needToDraw.emit()

        else:
            assert self.setup_name is not None
            self.add_str_to_list(
                self.setup_name
                + ": "
                + ",".join(
                    [
                        f"[{x:.1f}, {y:.1f}]"
                        for x, y in self.setup_points_dict[self.setup_name][1]
                    ]
                ),
                color=self.setup_points_dict[self.setup_name][0],
            )
            self.setup_name = None

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.setup_points_dict.pop(item.data(Qt.ItemDataRole.UserRole).split(":")[0])
        self.list.takeItem(self.list.row(item))
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
            self.color_count = (
                0 if self.color_count == n_colors - 1 else self.color_count + 1
            )
            name, points_str = value.split(":")
            list_of_points = literal_eval(points_str)
            if len(list_of_points) == 2 and not isinstance(
                list_of_points[0], (list, tuple)
            ):
                # only one setup point
                list_of_points = [list_of_points]
            self.setup_points_dict[name] = (QColors[self.color_count], list_of_points)
            self.add_str_to_list(value, color=QColors[self.color_count])

    def paint_on_canvas(self, painter: CustomQPainter):
        if not self.CheckBox.isChecked():
            return

        painter.setPenColor(QColor("black"))  # set pen to color black
        for color, points in self.setup_points_dict.values():
            painter.setBrush(color)
            for point in points:
                painter.drawBigPoint(*point)
