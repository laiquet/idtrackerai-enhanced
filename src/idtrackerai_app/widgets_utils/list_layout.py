from PyQt6.QtWidgets import (
    QCheckBox,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QListWidget,
    QLabel,
    QWidget,
    QListWidgetItem,
)

from PyQt6.QtCore import Qt, QSize, pyqtSignal
import numpy as np


class ListLayout(QVBoxLayout):
    ListChanged = pyqtSignal()
    draw_and_flush = pyqtSignal()

    def __init__(self, name=""):
        super().__init__()
        self.CheckBox = QCheckBox(name)
        self.CheckBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.CheckBox.stateChanged.connect(self.CheckBox_changed)

        self.add = QPushButton("Add", visible=False)
        self.add.setCheckable(True)
        self.add.setFixedWidth(70)

        self.list = QListWidget(visible=False)
        self.list.setAlternatingRowColors(True)
        self.update_height()

        self.list.model().rowsInserted.connect(self.update_height)
        self.list.model().rowsRemoved.connect(self.update_height)

        Controls_HBox = QHBoxLayout()
        Controls_HBox.addWidget(self.CheckBox)
        Controls_HBox.addWidget(self.add)

        self.addLayout(Controls_HBox)
        self.addWidget(self.list)

        self.list.model().rowsInserted.connect(self.ListChanged.emit)
        self.list.model().rowsRemoved.connect(self.ListChanged.emit)

    def update_height(self):
        n_rows = max(2, min(5, self.list.count()))
        self.list.setFixedHeight(
            25 * n_rows + 2 * self.list.frameWidth(),
        )

    def CheckBox_changed(self, enabled):
        self.list.setVisible(enabled)
        self.add.setVisible(enabled)

    def enter_key_event(self):
        if self.add.isChecked():
            self.add.click()

    def str_list(self) -> str:
        if self.CheckBox.isChecked():
            return [
                self.list.item(i).data(Qt.UserRole)
                for i in range(self.list.count())
            ]

        else:
            return None

    def add_ax_reference(self, ax):
        self.ax = ax
        (self.plot_line,) = ax.plot([], [], ".")

    def click_event(self, event):
        xy = self.plot_line.get_xydata()
        self.plot_line.set_data(np.vstack([xy, (event.xdata, event.ydata)]).T)
        self.draw_and_flush.emit()

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.list.takeItem(self.list.row(item))

    def add_str_to_list(self, text: str):
        cw = CustomListItem(text, remove_func=self.remove_item)
        item = QListWidgetItem()
        item.setData(Qt.UserRole, text)
        item.setSizeHint(QSize(40, 25))
        self.list.addItem(item)
        self.list.setItemWidget(item, cw)


class CustomListItem(QWidget):
    def __init__(self, text, remove_func=None):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(11, 0, 11, 0)
        self.layout().addWidget(QLabel(text))

        rm_btn = QPushButton("Remove")
        rm_btn.setFixedSize(QSize(80, 20))
        rm_btn.clicked.connect(remove_func)
        self.layout().addWidget(rm_btn)
