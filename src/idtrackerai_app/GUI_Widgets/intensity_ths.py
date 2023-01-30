from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from idtrackerai_app.widgets_utils import LabeledSlider, LabelRangeSlider


class IntensityThresholds(QWidget):
    newValue = pyqtSignal(object)

    def __init__(self, parent, min, max):
        super().__init__()
        self.parent_widget = parent
        self.label = QLabel("Intensity thresholds")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range_slider = LabelRangeSlider(parent=parent, min=min, max=max)
        self.simple_slider = LabeledSlider(parent, min=min, max=max)
        self.simple_slider.setVisible(False)

        self.range_slider.valueChanged.connect(self.newValue.emit)
        self.simple_slider.valueChanged.connect(lambda x: self.newValue.emit((0, x)))
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.label)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.simple_slider)

    def bkg_changed(self, bkg):
        if bkg is None:
            self.label.setText("Intensity thresholds")
            self.range_slider.setVisible(True)
            self.simple_slider.setVisible(False)
            self.newValue.emit(self.range_slider.value())
        else:
            self.label.setText("Background difference threshold")
            self.range_slider.setVisible(False)
            self.simple_slider.setVisible(True)
            self.newValue.emit((0, self.simple_slider.value()))

    def setValue(self, value):
        self.range_slider.setValue(value)
        self.simple_slider.setValue(value[1])

    def value(self):
        if self.range_slider.isVisible():
            return self.range_slider.value()
        else:
            return (0, self.simple_slider.value())

    def setToolTip(self, tooltip: str):
        self.range_slider.setToolTip(tooltip)
        self.simple_slider.setToolTip(tooltip)
