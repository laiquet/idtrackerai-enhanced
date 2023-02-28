from math import sqrt
from re import compile

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QInputDialog, QToolButton, QVBoxLayout, QWidget

from idtrackerai_GUI_tools import CustomList, CustomPainter


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = compile(r"[@!$%^&*?/\~:|]")
    return regex.search(string) is not None


QColors = [
    0x9467BD,
    0x2CA02C,
    0xBCBD22,
    0xFF7F0E,
    0x8C564B,
    0xE377C2,
    0x7F7F7F,
    0x17BECF,
]
n_colors = len(QColors)


class SetupPoints(QWidget):
    needToDraw = pyqtSignal()

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
        self.setup_points_dict: dict[str, tuple[int, list[tuple[float, float]]]] = {}
        self.list.ListChanged.connect(self.needToDraw.emit)
        self.list.removedItem.connect(self.remove_item)
        self.color_count = -1
        self.setup_name = None

    def click_event(self, button: int, zoom: float, x: float, y: float):
        if self.setup_name is not None:
            points = self.setup_points_dict[self.setup_name][1]
            if button == Qt.MouseButton.LeftButton:
                # Add clicked point
                points.append((x, y))
            elif button == Qt.MouseButton.RightButton:
                # Remove nearest point
                if not points:
                    return
                distances = [sqrt((px - x) ** 2 + (py - y) ** 2) for px, py in points]
                index, dist = min(enumerate(distances), key=lambda x: x[1])
                if dist < 20 * zoom:  # 20 px threshold
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
                0 if self.color_count == n_colors - 1 else self.color_count + 1
            )

            self.setup_name = name
            self.setup_points_dict[name] = (QColors[self.color_count], [])
            self.needToDraw.emit()

        else:
            if self.setup_name is None:
                return
            self.list.add_str(
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

    def remove_item(self, data: str):
        self.setup_points_dict.pop(data.split(":")[0])

    def load_points(self, values: dict[str, list[tuple[float, float]]]):
        self.list.clear()
        self.setup_points_dict.clear()

        for name, points in values.items():
            self.color_count = (
                0 if self.color_count == n_colors - 1 else self.color_count + 1
            )
            self.setup_points_dict[name] = (QColors[self.color_count], points)
            self.list.add_str(
                name + ": " + ",".join([f"[{x:.1f}, {y:.1f}]" for x, y in points]),
                color=QColors[self.color_count],
            )

    def get_points(self) -> dict[str, list[tuple[float, float]]]:
        return {key: value[1] for key, value in self.setup_points_dict.items()}

    def paint_on_canvas(self, painter: CustomPainter):
        painter.setPenColor(0x000000)  # set pen to color black
        for color, points in self.setup_points_dict.values():
            painter.setBrush(color)
            for point in points:
                painter.drawBigPoint(*point)

    def enter_pressed(self):
        self.add.setChecked(False)
