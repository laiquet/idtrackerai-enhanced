from PyQt6.QtCore import Qt, pyqtSignal
from superqt import QLabeledDoubleRangeSlider

# TODO: Change by QLabeledRangeSlider when the PR is accepted
# (https://github.com/napari/superqt/pull/111)
class LabeleRangeSlider(QLabeledDoubleRangeSlider):
    has_changed = pyqtSignal()

    def __init__(self, min, max, start_val, end_val):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(int(min), int(max))
        self.setValue([int(start_val), int(end_val)])
        self.setFixedHeight(40)
        self._max_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._min_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderPressed.connect(self.has_changed.emit)
        self._slider.sliderReleased.connect(self.has_changed.emit)
