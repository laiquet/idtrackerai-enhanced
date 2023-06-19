from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from idtrackerai_GUI_tools import LabeledSlider, LabelRangeSlider


class IntensityThresholds(QWidget):
    newValue = Signal(object)

    def __init__(self, parent, min, max):
        super().__init__()
        self.parent_widget = parent
        self.label_nobkg = QLabel("Blob intensity\nthresholds")
        self.label_nobkg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_yesbkg = QLabel("Background\ndifference threshold")
        self.label_yesbkg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_yesbkg.setVisible(False)
        self.range_slider = LabelRangeSlider(parent=parent, min=min, max=max)
        self.simple_slider = LabeledSlider(parent, min=min, max=max)
        self.simple_slider.setVisible(False)

        self.range_slider.valueChanged.connect(self.newValue.emit)
        self.simple_slider.valueChanged.connect(lambda x: self.newValue.emit((0, x)))
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.label_nobkg)
        layout.addWidget(self.label_yesbkg)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.simple_slider)

    def bkg_changed(self, bkg):
        if bkg is None:
            self.label_nobkg.setVisible(True)
            self.label_yesbkg.setVisible(False)
            self.range_slider.setVisible(True)
            self.simple_slider.setVisible(False)
            self.newValue.emit(self.range_slider.value())
        else:
            self.label_nobkg.setVisible(False)
            self.label_yesbkg.setVisible(True)
            self.range_slider.setVisible(False)
            self.simple_slider.setVisible(True)
            self.newValue.emit((0, self.simple_slider.value()))

    def setValue(self, value):
        self.range_slider.setValue(value)
        self.simple_slider.setValue(value[1])

    def value(self):
        if self.range_slider.isVisible():
            return self.range_slider.value()
        return 0, self.simple_slider.value()

    def setToolTips(self, tooltip_nobkg: str, tooltip_yesbkg: str):
        self.label_nobkg.setToolTip(tooltip_nobkg)
        self.label_yesbkg.setToolTip(tooltip_yesbkg)
        self.range_slider.setToolTip(tooltip_nobkg)
        self.simple_slider.setToolTip(tooltip_yesbkg)
