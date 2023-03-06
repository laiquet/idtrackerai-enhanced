from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QPainterPath, QPalette, QResizeEvent
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QWidget
from superqt import QLabeledRangeSlider, QLabeledSlider

from idtrackerai.utils import get_vertices_from_label


class QHLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setContentsMargins(10, 0, 10, 0)
        self.setEnabled(False)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.EnabledChange:
            self.setEnabled(False)


def build_ROI_patches_from_list(width, height, list_of_ROIs) -> QPainterPath:
    path = QPainterPath()
    if not list_of_ROIs:
        return path

    if list_of_ROIs[0][0] == "+":
        path.addRect(0, 0, width, height)

    for line in list_of_ROIs:
        points = get_vertices_from_label(line)
        path_i = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path_i.lineTo(*point)

        if line[0] == "+":
            path -= path_i.simplified()
        elif line[0] == "-":
            path += path_i.simplified()
        else:
            raise TypeError
    return path


class LabeledSlider(QLabeledSlider):
    def __init__(self, parent: QWidget, min, max):
        self.parent_widget = parent
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(min, max)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        self._label.wheelEvent = lambda e: e.accept()

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.EnabledChange,
            QEvent.Type.FontChange,
        ):
            style = (
                "QDoubleSpinBox{"
                + f"color: #{self.palette().color(QPalette.ColorRole.Text).rgba():x}"
                f";background:transparent; border: 0; font-size:{self.font().pointSize()}pt"
                "}QDoubleSpinBox:!enabled{color: #"
                + f"{self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).rgba():x}"
                ";}"
            )
            self._label.setStyleSheet(style)
            self._label._update_size()


class LabelRangeSlider(QLabeledRangeSlider):
    def __init__(
        self,
        min,
        max,
        parent: QWidget | None = None,
        start_end_val=None,
        block_upper=True,
    ):
        self.parent_widget = parent
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(min, max)
        if start_end_val:
            self.setValue(start_end_val)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        self._min_label.wheelEvent = lambda e: e.ignore()
        self._max_label.wheelEvent = lambda e: e.ignore()
        for handle in self._handle_labels:
            handle.wheelEvent = lambda event: event.ignore()

        self._min_label.setReadOnly(True)
        if block_upper:
            self._max_label.setReadOnly(True)

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.EnabledChange,
            QEvent.Type.FontChange,
        ):
            style = (
                "QDoubleSpinBox{"
                + f"color: #{self.palette().color(QPalette.ColorRole.Text).rgba():x}"
                f";background:transparent; border: 0; font-size:{self.font().pointSize()}pt"
                "}QDoubleSpinBox:!enabled{color: #"
                + f"{self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).rgba():x}"
                ";}"
            )
            self._slider.setPalette(self.palette())
            self._min_label.setStyleSheet(style)
            self._max_label.setStyleSheet(style)
            self._max_label._update_size()
            self._min_label._update_size()
            for handle in self._handle_labels:
                handle.setStyleSheet(style)
                handle._update_size()


class WrappedLabel(QLabel):
    def __init__(
        self,
        text: str = "",
        framed: bool = False,
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
    ):
        super().__init__(text)
        if framed:
            self.setBackgroundRole(QPalette.ColorRole.Base)
            self.setAutoFillBackground(True)
            self.setContentsMargins(5, 3, 5, 3)
        self.setAlignment(align)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def set_size(self):
        self.setMinimumHeight(0)
        self.setMinimumHeight(max(self.heightForWidth(self.width()), 1))

    def resizeEvent(self, a0: QResizeEvent):
        self.set_size()
        super().resizeEvent(a0)

    def setText(self, text: str):
        # Add Zero-width space in backslashes for proper word wrapping
        super().setText(text.replace("\\", "\\\u200B"))
        self.set_size()

    def text(self):
        return super().text().replace("\u200B", "")


def key_event_modifier(event: QKeyEvent) -> QKeyEvent | None:
    if event.key() == Qt.Key.Key_W:
        return QKeyEvent(event.type(), Qt.Key.Key_Up, event.modifiers())
    if event.key() == Qt.Key.Key_S:
        return QKeyEvent(event.type(), Qt.Key.Key_Down, event.modifiers())
    if event.key() in (Qt.Key.Key_D, Qt.Key.Key_A, Qt.Key.Key_Left, Qt.Key.Key_Right):
        # These keys would be accepted by QTableWidget
        # but we want them to control the VideoPlayer
        event.ignore()
        return None
    return event
