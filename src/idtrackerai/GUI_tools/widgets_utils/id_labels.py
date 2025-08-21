import random

from qtpy.QtCore import Qt, Signal  # pyright: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QColor
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

from .other_utils import get_icon


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
        self._default_colors = None  # Store default colors

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

        self._default_colors = [
            QColor.fromHsv(int(360 * (h / len(labels))), 255, 255)
            for h in range(len(labels))
        ]

        if colors is None:
            colors = self._default_colors

        self.reshuffle_btn = QPushButton()
        self.reshuffle_btn.setIcon(get_icon("refresh"))
        self.reshuffle_btn.setText("Shuffle")
        self.reshuffle_btn.clicked.connect(self.reshuffle_colors)

        self.reset_btn = QPushButton()
        self.reset_btn.setIcon(get_icon("undo"))
        self.reset_btn.setText("Reset")
        self.reset_btn.clicked.connect(self.reset_colors)

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

        self.grid_layout.addWidget(self.reshuffle_btn, len(labels) + 2, 2)
        self.grid_layout.addWidget(self.reset_btn, len(labels) + 3, 2)

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

    def set_labels_enabled(self, enabled: bool) -> None:
        """Enable or disable label editing."""
        for idx in range(1, len(self.labels)):
            label_edit = self.grid_layout.itemAtPosition(idx + 1, 1)
            if label_edit is not None:
                edit_widget = label_edit.widget()
                if isinstance(edit_widget, QLineEdit):
                    edit_widget.setEnabled(enabled)

    def labels_are_enabled(self) -> bool:
        """Check if label editing is enabled."""
        for idx in range(1, len(self.labels)):
            label_edit = self.grid_layout.itemAtPosition(idx + 1, 1)
            if label_edit is not None:
                edit_widget = label_edit.widget()
                if isinstance(edit_widget, QLineEdit) and not edit_widget.isEnabled():
                    return False
        return True

    def colors_are_enabled(self) -> bool:
        """Check if color buttons are enabled."""
        for idx in range(1, len(self.colors)):
            color_btn = self.grid_layout.itemAtPosition(idx + 1, 2)
            if color_btn is not None:
                btn_widget = color_btn.widget()
                if isinstance(btn_widget, QPushButton) and not btn_widget.isEnabled():
                    return False
        return True

    def set_colors_enabled(self, enabled: bool) -> None:
        """Enable or disable color buttons."""
        for idx in range(len(self.colors)):
            color_btn = self.grid_layout.itemAtPosition(idx + 1, 2)
            if color_btn is not None:
                btn_widget = color_btn.widget()
                if isinstance(btn_widget, QPushButton):
                    btn_widget.setEnabled(enabled)
        self.reshuffle_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)

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

    def reset_colors(self):
        """Reset colors to their default values and update the UI."""
        if not self._default_colors:
            return
        self.colors = [QColor(255, 255, 255)] + self._default_colors
        self.transparent_colors = [
            QColor(c.red(), c.green(), c.blue(), 77) for c in self.colors
        ]

        for idx, color in enumerate(self.colors):
            row = idx + 1  # row 0 is the buttons
            color_btn = self.grid_layout.itemAtPosition(row, 2)
            if color_btn is not None:
                btn_widget = color_btn.widget()
                if isinstance(btn_widget, QPushButton):
                    set_button_color(btn_widget, color)
        self.needToDraw.emit()


def set_button_color(button: QPushButton, color: QColor):
    """Set the background color of a QPushButton."""
    unsaturated_color = QColor(color)
    unsaturated_color.setHsv(
        unsaturated_color.hue(),
        int(unsaturated_color.saturation() * 0.5),
        unsaturated_color.value(),
        100,
    )
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
        QPushButton:disabled {{
            background-color: rgba({unsaturated_color.red()}, {unsaturated_color.green()}, {unsaturated_color.blue()}, {unsaturated_color.alpha()});
            color: gray;
            border: gray;
        }}
        """
    )
    button.setText(color.name())
