from typing import Optional

import numpy as np
from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QKeyEvent, QPainterPath, QPalette, QResizeEvent
from qtpy.QtWidgets import QFrame, QLabel, QSizePolicy, QWidget
from superqt import QLabeledRangeSlider, QLabeledSlider
from superqt.sliders._labeled import LabelPosition

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


def build_ROI_patches_from_list(
    list_of_ROIs: list[str] | None, resolution_reduction: float, width: int, height: int
) -> QPainterPath:
    path = QPainterPath()

    if list_of_ROIs is None:
        return path

    path.addRect(
        -0.5, -0.5, width * resolution_reduction, height * resolution_reduction
    )

    for line in list_of_ROIs:
        points = (get_vertices_from_label(line) * resolution_reduction + 0.5).astype(
            np.int32
        )
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
        self._label.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self._label.valueChanged.connect(lambda val: self._slider.setValue(int(val)))

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
                ";background:transparent; border: 0;"
                f" font-size:{self.font().pointSize()}pt"
                "}QDoubleSpinBox:!enabled{color: #"
                + f"{self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).rgba():x}"
                ";}"
            )
            self._label.setStyleSheet(style)
            self._label._update_size()


class LabelRangeSlider(QLabeledRangeSlider):
    def __init__(
        self,
        min: int,
        max: int,
        parent: Optional[QWidget] = None,
        start_end_val: Optional[tuple[int, int]] = None,
        block_upper=True,
    ):
        self.parent_widget = parent
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.block_upper = block_upper
        self.setRange(min, max)
        self.setValue(start_end_val or (min, max))
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        self._min_label.setReadOnly(True)
        if block_upper:
            self._max_label.setReadOnly(True)
        else:
            self._max_label.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        self._handle_labels[0].valueChanged.connect(
            lambda val: self._slider.setSliderPosition(int(val), 0)
        )
        self._handle_labels[1].valueChanged.connect(
            lambda val: self._slider.setSliderPosition(int(val), 1)
        )

        for handle in self._handle_labels:
            handle.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

    def _reposition_labels(self):
        """Overriding superqt method to remove the last label.clearFocus() call"""
        if (
            not self._handle_labels
            or self._handle_label_position == LabelPosition.NoLabel
        ):
            return

        horizontal = self.orientation() == Qt.Orientation.Horizontal
        labels_above = self._handle_label_position == LabelPosition.LabelsAbove

        last_edge = None
        for i, label in enumerate(self._handle_labels):
            rect = self._slider._handleRect(i)
            dx = -label.width() / 2
            dy = -label.height() / 2
            if labels_above:
                if horizontal:
                    dy *= 3
                else:
                    dx *= -1
            else:
                if horizontal:
                    dy *= -1
                else:
                    dx *= 3
            pos = self._slider.mapToParent(rect.center())
            pos += QPoint(int(dx + self.label_shift_x), int(dy + self.label_shift_y))
            if last_edge is not None:
                # prevent label overlap
                if horizontal:
                    pos.setX(int(max(pos.x(), last_edge.x() + label.width() / 2 + 12)))
                else:
                    pos.setY(int(min(pos.y(), last_edge.y() - label.height() / 2 - 4)))
            label.move(pos)
            last_edge = pos
            # label.clearFocus() # better focus behavior without this
            label.show()
        self.update()

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
                ";background:transparent; border: 0;"
                f" font-size:{self.font().pointSize()}pt"
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

    def value(self) -> tuple[int, int]:
        return super().value()  # type: ignore

    def setValue(self, value) -> None:
        if not self.block_upper:
            self.setMaximum(value[1])
        return super().setValue(value)


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
        super().setText(text.replace("\\", "\\\u200b"))
        self.set_size()

    def text(self):
        return super().text().replace("\u200b", "")


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
