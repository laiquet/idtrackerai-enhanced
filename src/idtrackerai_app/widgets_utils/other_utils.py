from PyQt6.QtCore import Qt, pyqtSignal
from superqt import QLabeledDoubleRangeSlider
from PyQt6.QtWidgets import QLabel

# TODO: Change by QLabeledRangeSlider when the PR is accepted
# (https://github.com/napari/superqt/pull/111)
class LabelRangeSlider(QLabeledDoubleRangeSlider):
    has_changed = pyqtSignal()

    def __init__(self, min, max, start_end_val=None):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(int(min), int(max))
        if start_end_val:
            self.setValue(start_end_val)
        self.setFixedHeight(40)
        self._max_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._min_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderPressed.connect(self.has_changed.emit)
        self._slider.sliderReleased.connect(self.has_changed.emit)


class WrappedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self.setMaximumHeight(self.heightForWidth(self.width()))
