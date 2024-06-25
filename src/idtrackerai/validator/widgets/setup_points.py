from re import compile

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QColor, QColorConstants
from qtpy.QtWidgets import QInputDialog, QToolButton, QVBoxLayout, QWidget

from idtrackerai.GUI_tools import (
    CanvasMouseEvent,
    CanvasPainter,
    CustomList,
    point_colors,
)


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = compile(r"[@!$%^&*?/\~:|]")
    return regex.search(string) is not None


class SetupPoints(QWidget):
    needToDraw = Signal()

    def __init__(self):
        super().__init__()

        self.add = QToolButton()
        self.add.setText("Add")
        self.add.setCheckable(True)

        self.list = CustomList(max_n_row=0)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        self.setLayout(layout)
        layout.addWidget(self.add)
        layout.addWidget(self.list)

        self.add.toggled.connect(self.add_clicked)
        self.setup_points_dict: dict[str, tuple[QColor, list[tuple[int, int]]]] = {}
        self.list.ListChanged.connect(self.needToDraw.emit)
        self.list.removedItem.connect(self.remove_item)
        self.color_count = -1
        self.setup_name = None

    def click_event(self, event: CanvasMouseEvent):
        if not self.isVisible() or not self.isEnabled() or self.setup_name is None:
            return
        points = self.setup_points_dict[self.setup_name][1]
        if event.button == Qt.MouseButton.LeftButton:
            # Add clicked point
            points.append(event.int_xy_data)
        elif event.button == Qt.MouseButton.RightButton:
            # Remove nearest point
            if not points:
                return
            distances = map(event.distance_to, points)
            index, dist = min(enumerate(distances), key=lambda x: x[1])
            if dist < 20 * event.zoom:  # 20 px threshold
                points.pop(index)
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
                    return
                name = name.strip()
                invalid = has_invalid_chars(name)
                if invalid:
                    dialog_text = "Invalid characters encountered"
                if name in existing_names:
                    invalid = True
                    dialog_text = "Enter a non existing name"
            self.color_count = (
                0 if self.color_count == len(point_colors) - 1 else self.color_count + 1
            )

            self.setup_name = name
            self.setup_points_dict[name] = (QColor(point_colors[self.color_count]), [])
            self.needToDraw.emit()

        else:
            if self.setup_name is None:
                return
            self.list.add_str(
                self.setup_name
                + ": "
                + ",".join(
                    [
                        f"[{x:d}, {y:d}]"
                        for x, y in self.setup_points_dict[self.setup_name][1]
                    ]
                ),
                color=self.setup_points_dict[self.setup_name][0],
            )
            self.setup_name = None

    def remove_item(self, data: str):
        self.setup_points_dict.pop(data.split(":")[0])

    def load_points(self, values: dict[str, list[tuple[int, int]]]):
        self.list.clear()
        self.setup_points_dict.clear()
        if not isinstance(values, dict):
            return  # Loading from v4

        for name, points in values.items():
            self.color_count = (
                0 if self.color_count == len(point_colors) - 1 else self.color_count + 1
            )
            self.setup_points_dict[name] = (
                QColor(point_colors[self.color_count]),
                points,
            )
            self.list.add_str(
                name
                + ": "
                + ",".join([f"[{int(x):d}, {int(y):d}]" for x, y in points]),
                color=QColor(point_colors[self.color_count]),
            )

    def get_points(self) -> dict[str, list[tuple[int, int]]]:
        return {key: value[1] for key, value in self.setup_points_dict.items()}

    def paint_on_canvas(self, painter: CanvasPainter):
        painter.setPenColor(QColorConstants.Black)
        for color, points in self.setup_points_dict.values():
            painter.setBrush(color)
            for point in points:
                painter.drawBigPoint(*point)
