import random

from qtpy.QtCore import Qt, Signal  # pyright: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QColor, QIcon
from qtpy.QtWidgets import (
    QApplication,
    QColorDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QWidget,
)


class IdLabels(QScrollArea):
    needToDraw = Signal()
    labels: list[str]

    def __init__(self):
        super().__init__()
        self.grid_layout = QGridLayout()
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidgetResizable(True)
        wid = QWidget()
        wid.setLayout(self.grid_layout)
        self.setWidget(wid)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def change_color(self, idx, btn: QPushButton):
        color = QColorDialog.getColor(
            btn.palette().button().color(),
            self,
            options=QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if not color.isValid():
            return
        set_button_color(btn, color)
        self.colors[idx] = color
        self.transparent_colors[idx] = QColor(
            color.red(), color.green(), color.blue(), 77
        )

    def load(self, labels: list[str], colors: list[str] | list[QColor] | None = None):
        # Remove all widgets from the grid layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if colors is None:
            colors = [
                QColor.fromHsv(int(360 * (h / len(labels))), 255, 255)
                for h in range(len(labels))
            ]

        # Add "Reshuffle colors" button
        reshuffle_btn = QPushButton()
        reshuffle_btn.setIcon(QIcon.fromTheme("view-refresh"))
        reshuffle_btn.setText("Shuffle")
        reshuffle_btn.clicked.connect(self.reshuffle_colors)
        self.grid_layout.addWidget(reshuffle_btn, 0, 1, 1, -1)

        color_btn = QPushButton()
        edit = QLabel("null")
        edit.setEnabled(False)
        edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_button_color(color_btn, QColor(255, 255, 255))
        color_btn.clicked.connect(
            lambda checked, btn=color_btn: self.change_color(0, btn)
        )

        self.grid_layout.addWidget(edit, 1, 1)
        self.grid_layout.addWidget(color_btn, 1, 2)

        self.labels = [""]
        self.colors = [QColor(255, 255, 255)]

        for indx, (label, color) in enumerate(zip(labels, colors), 1):
            if isinstance(color, str):
                color = QColor(color)
            edit = QLineEdit()
            edit.setText(label)
            edit.setPlaceholderText(str(indx))
            edit.setObjectName(str(indx))
            color_btn = QPushButton()
            set_button_color(color_btn, color)
            color_btn.clicked.connect(
                lambda checked, idx=indx, btn=color_btn: self.change_color(idx, btn)
            )
            color_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            color_btn.customContextMenuRequested.connect(
                lambda pos, btn=color_btn, idx=indx: self.show_color_menu(btn, idx, pos)
            )
            edit.textChanged.connect(self.new_label)
            edit.editingFinished.connect(self.validate_label)
            self.grid_layout.addWidget(QLabel(f"{indx}:"), indx + 1, 0)
            self.grid_layout.addWidget(edit, indx + 1, 1)
            self.grid_layout.addWidget(color_btn, indx + 1, 2)

            self.labels.append(label)
            self.colors.append(color)

        self.transparent_colors = [
            QColor(color.red(), color.green(), color.blue(), 77)
            for color in self.colors
        ]

    def show_color_menu(self, btn: QPushButton, idx: int, pos):
        """Show context menu to copy color code."""
        menu = QMenu(btn)
        hex_code = self.colors[idx].name()
        copy_action = menu.addAction(f"Copy {hex_code}")
        assert copy_action is not None
        copy_action.triggered.connect(
            lambda: QApplication.clipboard().setText(hex_code)
        )
        menu.exec(btn.mapToGlobal(pos))

    def validate_label(self):
        sender = self.sender()
        assert isinstance(sender, QLineEdit)
        text = sender.text()
        if not text:
            sender.setText(sender.placeholderText())
        else:
            sender.setText(text.strip())

    def new_label(self, new_label=""):
        self.labels[int(self.sender().objectName())] = new_label
        self.needToDraw.emit()

    def set_colors_enabled(self, enabled: bool) -> None:
        """Enable or disable color buttons."""
        for idx in range(1, len(self.colors)):
            color_btn = self.grid_layout.itemAtPosition(idx + 1, 2)
            if color_btn is not None:
                btn_widget = color_btn.widget()
                if isinstance(btn_widget, QPushButton):
                    btn_widget.setEnabled(enabled)

    def get_labels(self) -> list[str]:
        return self.labels

    def get_colors(self) -> tuple[list[QColor], list[QColor]]:
        return self.colors, self.transparent_colors

    def reshuffle_colors(self) -> None:
        """Reshuffle the existing colors among the labels and update the UI."""
        if not hasattr(self, "colors") or not self.colors:
            return

        # Do not reshuffle the first color (index 0), which is for 'null'
        label_colors = self.colors[1:]
        random.shuffle(label_colors)
        self.colors[1:] = label_colors
        self.transparent_colors = [
            QColor(c.red(), c.green(), c.blue(), 77) for c in self.colors
        ]

        # Update color buttons in the grid layout
        for idx, color in enumerate(self.colors):
            row = idx + 1  # row 0 is the shuffle button
            color_btn = self.grid_layout.itemAtPosition(row, 2)
            if color_btn is not None:
                btn_widget = color_btn.widget()
                if isinstance(btn_widget, QPushButton):
                    set_button_color(btn_widget, color)

        self.needToDraw.emit()


def set_button_color(button: QPushButton, color: QColor):
    """Set the background color of a QPushButton."""
    button.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {color.name()};
            border: black;
            border-width: 1px;
            border-style: solid;
            border-radius: 5px;
            padding: 5px;
            color: black;
        }}
        QPushButton:hover {{
            background-color: {color.lighter(170).name()};
        }}
        """
    )
    button.setText(color.name())
