from PyQt6.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QToolButton,
)


class ListLayout(QWidget):
    valueChanged = pyqtSignal(object)
    needToDraw = pyqtSignal()
    ListChanged = pyqtSignal()
    newItemSelected = pyqtSignal(object)

    def __init__(self, parent, name=""):
        self.parent_widget = parent
        super().__init__()
        self.CheckBox = QCheckBox(name)
        self.CheckBox.stateChanged.connect(self.CheckBox_changed)
        self.CheckBox.stateChanged.connect(self.ListChanged.emit)

        self.add = QToolButton()
        self.add.setText("Add")
        self.add.setCheckable(True)

        self.list = _QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.lost_focus.connect(self.list_lost_focus)
        self.update_height()

        self.ListChanged.connect(self.update_height)
        self.list.model().rowsInserted.connect(self.ListChanged.emit)
        self.list.model().rowsRemoved.connect(self.ListChanged.emit)
        self.list.itemPressed.connect(self.item_selected)
        self.list.currentItemChanged.connect(lambda x, y: self.item_selected(x))
        self.selected_item = None

        Controls_HBox = QHBoxLayout()
        Controls_HBox.addWidget(self.CheckBox)
        Controls_HBox.addWidget(self.add)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.setLayout(layout)
        layout.addLayout(Controls_HBox)
        layout.addWidget(self.list)

        self.list.installEventFilter(self)
        self.CheckBox_changed(False)

    def update_height(self):
        n_rows = max(2, min(5, self.list.count()))
        self.list.setFixedHeight(25 * n_rows + 2 * self.list.frameWidth())

    def CheckBox_changed(self, enabled):
        if self.add.isChecked():
            self.add.click()
        self.list.setVisible(enabled)
        self.add.setVisible(enabled)

    def enter_key_event(self):
        if self.add.isChecked():
            self.add.click()

    def getValue(self) -> list[str]:
        if self.CheckBox.isChecked():
            return [
                self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())
            ]

        else:
            return None

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.item_selected(None)
        self.list.takeItem(self.list.row(item))
        self.list.clearFocus()

    def add_str_to_list(self, text: str, color: QColor | None = None):
        cw = CustomListItem(
            text, remove_func=self.remove_item, parent=self.parent_widget, color=color
        )
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, text)
        item.setSizeHint(QSize(40, 25))
        self.list.addItem(item)
        self.list.setItemWidget(item, cw)
        self.add.clearFocus()

    def item_selected(self, item: QListWidgetItem):
        if self.selected_item == item:
            return
        if self.selected_item is not None:
            widget = self.list.itemWidget(self.selected_item)
            if widget is not None:
                widget.lost_focus()

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
    def __init__(
        self, text, parent: QWidget, remove_func=None, color: None | QColor = None
    ):
        super().__init__(parent)
        self.selected = False
        self.text = QLabel(text)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(11, 0, 11, 0)

        if color is not None:
            icon = QLabel()
            icon.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            icon.setFixedSize(10, 10)
            pixmap = QPixmap(icon.size())
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(pixmap.rect())
            painter.end()
            icon.setPixmap(pixmap)
            self.layout().addWidget(icon)

        rm_btn = QToolButton()
        rm_btn.setText("Remove")
        rm_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rm_btn.clicked.connect(remove_func)
        self.layout().addWidget(self.text)
        self.layout().addWidget(rm_btn)
        self.text.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rm_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.update_label_colors()

    def gain_focus(self):
        self.selected = True
        self.text.setStyleSheet(self.selected_stylesheet)

    def lost_focus(self):
        self.selected = False
        self.text.setStyleSheet(self.non_selected_stylesheet)

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self.update_label_colors()

    def update_label_colors(self):
        self.selected_stylesheet = (
            "QLabel {color : #"
            + f"{self.palette().highlightedText().color().rgb():x}"
            + "; }"
        )
        self.non_selected_stylesheet = (
            "QLabel {color : #" + f"{self.palette().text().color().rgb():x}" + "; }"
        )
        if self.selected:
            self.text.setStyleSheet(self.selected_stylesheet)
        else:
            self.text.setStyleSheet(self.non_selected_stylesheet)
