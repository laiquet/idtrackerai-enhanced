# Each Qt binding is different, so...
# pyright: reportIncompatibleMethodOverride=false
import logging
from collections.abc import Sequence

from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QSlider, QSpinBox, QWidget
from superqt import QLabeledRangeSlider


class LabelRangeSlider(QLabeledRangeSlider):
    single_value_changed = Signal(int)

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

        def wheelEvent_on_min(event: QWheelEvent) -> None:
            steps = event.angleDelta().y() // 120
            val0, val1 = self._slider.value()
            self._slider.setSliderPosition((val0 + steps, val1))

        def wheelEvent_on_max(event: QWheelEvent) -> None:
            steps = event.angleDelta().y() // 120
            val0, val1 = self._slider.value()
            self._slider.setSliderPosition((val0, val1 + steps))

        self._handle_labels[0].wheelEvent = wheelEvent_on_min
        self._handle_labels[1].wheelEvent = wheelEvent_on_max

        def send_single_value_changed(value: int) -> None:
            """Emit the single value changed signal for the first handle"""
            try:
                self.single_value_changed.emit(int(value))
            except Exception as e:
                logging.error(f"Error emitting single_value_changed signal: {e}")

        for handle in self._handle_labels:
            handle.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
            handle.textChanged.connect(send_single_value_changed)

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
        # We manage wheel events in this class
        self.slider.wheelEvent = lambda e: e.ignore() if e is not None else None
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

    def wheelEvent(self, e: QWheelEvent) -> None:
        steps = e.angleDelta().y()
        if steps > 0:
            self.setValue(self.value() + 1)
        elif steps < 0:
            self.setValue(self.value() - 1)
        else:
            e.ignore()
        e.accept()
