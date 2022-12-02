from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ListLayout(QVBoxLayout):
    ListChanged = pyqtSignal()
    update_player = pyqtSignal(bool)
    newItemSelected = pyqtSignal(object)

    def __init__(self, parent, name=""):
        self.parent = parent
        super().__init__()
        self.CheckBox = QCheckBox(name)
        self.CheckBox.stateChanged.connect(self.CheckBox_changed)

        self.add = QPushButton("Add", visible=False)
        # self.add.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.add.setCheckable(True)
        self.add.setFixedWidth(70)

        self.list = _QListWidget(visible=False)
        self.list.setAlternatingRowColors(True)
        self.list.lost_focus.connect(self.list_lost_focus)
        self.update_height()

        self.ListChanged.connect(self.update_height)
        self.list.model().rowsInserted.connect(self.ListChanged.emit)
        self.list.model().rowsRemoved.connect(self.ListChanged.emit)
        self.list.itemClicked.connect(self.item_selected)
        self.list.currentItemChanged.connect(
            lambda x, y: self.item_selected(x)
        )
        self.selected_item = None

        Controls_HBox = QHBoxLayout()
        Controls_HBox.addWidget(self.CheckBox)
        Controls_HBox.addWidget(self.add)

        self.addLayout(Controls_HBox)
        self.addWidget(self.list)

        self.list.installEventFilter(self)

    def update_height(self):
        n_rows = max(2, min(5, self.list.count()))
        self.list.setFixedHeight(
            25 * n_rows + 2 * self.list.frameWidth(),
        )

    def CheckBox_changed(self, enabled):
        self.list.setVisible(enabled)
        self.add.setVisible(enabled)
        self.ListChanged.emit()
        self.update_player.emit(False)

    def enter_key_event(self):
        if self.add.isChecked():
            self.add.click()

    def getValue(self) -> list[str]:
        if self.CheckBox.isChecked():
            return [
                self.list.item(i).data(Qt.UserRole)
                for i in range(self.list.count())
            ]

        else:
            return None

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.item_selected(None)
        self.list.takeItem(self.list.row(item))
        self.list.clearFocus()

    def add_str_to_list(self, text: str):
        cw = CustomListItem(
            text, remove_func=self.remove_item, parent=self.parent
        )
        item = QListWidgetItem()
        item.setData(Qt.UserRole, text)
        item.setSizeHint(QSize(40, 25))
        self.list.addItem(item)
        self.list.setItemWidget(item, cw)
        self.add.clearFocus()

    def item_selected(self, item: QListWidgetItem):
        if self.selected_item == item:
            return
        if self.selected_item is not None:
            self.list.itemWidget(self.selected_item).lost_focus()

        if item is not None:
            self.list.itemWidget(item).gain_focus()
        self.newItemSelected.emit(item)
        self.selected_item = item

    def list_lost_focus(self):
        self.list.clearSelection()
        item = self.list.itemWidget(self.selected_item)
        if item:
            item.lost_focus()
        self.newItemSelected.emit(None)
        self.selected_item = None


# TODO clean subclasses
class _QListWidget(QListWidget):
    lost_focus = pyqtSignal()

    def focusOutEvent(self, event):
        self.lost_focus.emit()
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearFocus()


class CustomListItem(QWidget):
    def __init__(self, text, parent: QWidget, remove_func=None):
        super().__init__()
        std_color = parent.palette().windowText().color().name()
        focus_color = parent.palette().highlightedText().color().name()

        self.std_style = "QLabel {color : " + std_color + "; }"
        self.focus_style = "QLabel {color : " + focus_color + "; }"
        self.text = QLabel(text)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(11, 0, 11, 0)
        self.layout().addWidget(self.text)

        rm_btn = QPushButton("Remove")
        rm_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rm_btn.setFixedSize(QSize(80, 20))
        rm_btn.clicked.connect(remove_func)
        self.layout().addWidget(rm_btn)

    def gain_focus(self):
        self.text.setStyleSheet(self.focus_style)

    def lost_focus(self):
        self.text.setStyleSheet(self.std_style)
