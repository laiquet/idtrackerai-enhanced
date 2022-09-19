from PyQt6.QtWidgets import (
    QCheckBox,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QListWidget,
    QLabel,
    QWidget,
)

from PyQt6.QtCore import Qt
import numpy as np


class List_Layout:
    def __init__(self, active=False):
        self.is_list_active = active

        self.CheckBox = QCheckBox("", enabled=active)
        self.CheckBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.CheckBox.stateChanged.connect(self.CheckBox_changed)

        self.add = QPushButton("Add")
        self.add.setCheckable(True)

        self.remove = QPushButton("Remove selected")
        self.remove.clicked.connect(self.remove_event)

        self.list = QListWidget()
        self.list.addItem("text just to fit list Height")
        self.list.setFixedHeight(
            self.list.sizeHintForRow(0) * 5 + 2 * self.list.frameWidth(),
        )
        self.list.clear()

        self.list.itemClicked.connect(
            lambda: self.remove.setEnabled(
                self.CheckBox.isChecked() and len(self.list.selectedItems())
            )
        )

        self.Main_Layout = QVBoxLayout()
        Controls_HBox = QHBoxLayout()
        Controls_HBox.addWidget(self.CheckBox)
        Controls_HBox.addWidget(self.add)
        Controls_HBox.addWidget(self.remove)

        self.Main_Layout.addLayout(Controls_HBox)
        self.Main_Layout.addWidget(self.list)

        self.CheckBox_changed(enabled=self.CheckBox.isChecked())

    def remove_event(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        if not len(self.list.selectedItems()):
            self.remove.setEnabled(False)

    def CheckBox_changed(self, enabled):
        self.list.setVisible(enabled)
        self.add.setEnabled(enabled)
        self.remove.setEnabled(enabled and len(self.list.selectedItems()))

    def set_enabled(self, enabled):
        self.CheckBox.setEnabled(enabled)
        self.list.setEnabled(enabled)
        self.remove.setEnabled(
            self.CheckBox.isChecked() and len(self.list.selectedItems())
        )
        self.add.setEnabled(self.CheckBox.isChecked())

    def enter_key_event(self):
        if self.add.isChecked():
            self.add.click()

    def str_list(self) -> str:
        if self.CheckBox.isChecked():
            return "\n".join(
                self.list.item(i).text() for i in range(self.list.count())
            )
        else:
            return None

    def add_ax_reference(self, ax):
        self.ax = ax
        (self.plot_line,) = ax.plot([], [], ".")

    def click_event(self, event):
        xy = self.plot_line.get_xydata()
        self.plot_line.set_data(np.vstack([xy, (event.xdata, event.ydata)]).T)
