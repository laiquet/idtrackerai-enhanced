from typing import Sequence

from qtpy.QtCore import QEvent, QPoint, Qt, Signal  # type: ignore
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QSlider, QSpinBox, QWidget
from superqt import QLabeledRangeSlider
from superqt.sliders._labeled import LabelPosition


class LabelRangeSlider(QLabeledRangeSlider):
    def __init__(
        self,
        min: int,
        max: int,
        parent: QWidget | None = None,
        start_end_val: tuple[int, int] | None = None,
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

    def setValue(self, value: Sequence[int]) -> None:
        if not self.block_upper and value[1] > self.maximum():
            self.setMaximum(value[1])
        return super().setValue(value)  # type: ignore


class InvertibleSlider(QWidget):
    "A labeled slider with the capacity to invert the colored bar without negative numbers"

    valueChanged = Signal(int)

    def __init__(self, min: int, max: int) -> None:
        super().__init__()
        main_layout = QHBoxLayout()
        self.min = min
        self.max = max

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setRange(min, max)

        self.label = QSpinBox()
        self.label.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.label.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.label.setStyleSheet("background:transparent; border: 0;")
        self.label.setRange(min, max)

        main_layout.addWidget(self.slider)
        main_layout.addWidget(self.label)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        self.inverted = False

        self.slider.valueChanged.connect(self.slider_changed)
        self.label.valueChanged.connect(self.label_changed)

    def slider_changed(self, value: int):
        self.label.setValue(-value if self.inverted else value)

    def label_changed(self, value: int):
        self.slider.setValue(-value if self.inverted else value)
        self.valueChanged.emit(value)

    def value(self):
        return self.label.value()

    def setValue(self, val: int):
        return self.label.setValue(val)

    def set_inverted(self, inverted: bool):
        """This function must be followed by a valueChanged signal
        I don't implement it here to avoid duplicates and simplify code downstream"""
        self.slider.setInvertedAppearance(inverted)
        self.slider.setInvertedControls(inverted)
        current_value = self.slider.value()
        self.inverted = inverted

        self.slider.blockSignals(True)
        if inverted:
            self.slider.setRange(-self.max, self.min)
            self.slider.setValue(-abs(current_value))
        else:
            self.slider.setRange(self.min, self.max)
            self.slider.setValue(abs(current_value))
        self.slider.blockSignals(False)

    def set_value(self, value: int):
        self.label.setValue(value)
