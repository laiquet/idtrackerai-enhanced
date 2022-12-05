from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel
from superqt import QLabeledRangeSlider


class LabelRangeSlider(QLabeledRangeSlider):
    newValue = pyqtSignal()

    def __init__(self, min, max, start_end_val=None):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(min, max)
        if start_end_val:
            self.setValue(start_end_val)
        self.setFixedHeight(40)
        self._max_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._min_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderPressed.connect(self.newValue.emit)
        self._slider.sliderReleased.connect(self.newValue.emit)
        self.editingFinished.connect(self.newValue.emit)
        # self.valueChanged.connect(self.newValue.emit)


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
