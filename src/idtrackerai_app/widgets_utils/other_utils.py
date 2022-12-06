from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel
from superqt import QLabeledRangeSlider


class LabelRangeSlider(QLabeledRangeSlider):
    newValue = pyqtSignal(object)

    def __init__(self, min, max, start_end_val=None):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(min, max)
        if start_end_val:
            self.setValue(start_end_val)
        self.setFixedHeight(40)
        self._max_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._min_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderPressed.connect(
            lambda: self.newValue.emit(self.value())
        )
        self._slider.sliderReleased.connect(
            lambda: self.newValue.emit(self.value())
        )
        self.editingFinished.connect(lambda: self.newValue.emit(self.value()))
        # self.valueChanged.connect(self.newValue.emit)

    def setValue(self, value) -> None:
        super().setValue(value)
        self.newValue.emit(self.value())

    def value(self) -> list[int]:
        return list(super().value())


class WrappedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self.setMaximumHeight(self.heightForWidth(self.width()))

    def setText(self, text):
        # Add Zero-width space in backslashes for proper word wrapping
        super().setText(text.replace("\\", "\\\u200B"))

    def text(self):
        output = super().text()
        return output.replace("\u200B", "")
