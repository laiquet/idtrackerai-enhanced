from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel
from superqt import QLabeledRangeSlider, QLabeledDoubleRangeSlider


class my_QLabeleRangeSlider(QHBoxLayout):
    def __init__(self, name, min, max, start_val, end_val, func_to_connect):
        super().__init__()
        self.qlabel = QLabel(name)
        self.addWidget(self.qlabel)
        # Change by QLabeledRangeSlider when the PR is accepted (https://github.com/napari/superqt/pull/111)
        self.slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min), int(max))
        self.slider.setValue([int(start_val), int(end_val)])
        self.slider.setFixedHeight(40)
        self.slider._max_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider._min_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider._slider.sliderPressed.connect(func_to_connect)
        self.slider._slider.sliderReleased.connect(func_to_connect)
        self.addWidget(self.slider)

    def setEnabled(self, enabled):
        self.qlabel.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def value(self):
        return self.slider.value()
