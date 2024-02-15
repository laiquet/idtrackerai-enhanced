from typing import Sequence

from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from idtrackerai.GUI_tools import InvertibleSlider, LabelRangeSlider


class IntensityThresholds(QWidget):
    newValue = Signal(tuple)

    def __init__(self, parent, min, max):
        super().__init__()
        self.parent_widget = parent
        self.label_nobkg = QLabel("Blob intensity\nthresholds")
        self.label_nobkg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_yesbkg = QLabel("Background\ndifference threshold")
        self.label_yesbkg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_yesbkg.setVisible(False)
        self.range_slider = LabelRangeSlider(parent=parent, min=min, max=max)
        self.simple_slider = InvertibleSlider(min=min, max=max)
        self.range_slider.setVisible(False)
        self.th_type = QComboBox()
        self.th_type.addItems(["Dark animals", "Bright animals", "Custom"])
        self.th_type.currentTextChanged.connect(self.threshold_type_changed)
        self.simple_slider.valueChanged.connect(
            lambda: self.newValue.emit(self.value())
        )
        self.range_slider.valueChanged.connect(self.newValue.emit)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.label_nobkg)
        layout.addWidget(self.label_yesbkg)
        layout.addWidget(self.th_type)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.simple_slider)

    def threshold_type_changed(self, new_type: str):
        if new_type == "Dark animals":
            self.range_slider.setVisible(False)
            self.simple_slider.setVisible(True)
            self.simple_slider.set_inverted(False)
        elif new_type == "Bright animals":
            self.range_slider.setVisible(False)
            self.simple_slider.setVisible(True)
            self.simple_slider.set_inverted(True)
        elif new_type == "Custom":
            self.simple_slider.setVisible(False)
            self.range_slider.setVisible(True)
        elif new_type == "using_background":
            self.range_slider.setVisible(False)
            self.simple_slider.setVisible(True)
            self.simple_slider.set_inverted(True)
        else:
            raise ValueError(new_type)
        self.newValue.emit(self.value())

    def bkg_changed(self, bkg):  # bkg is np.ndarray of None
        if bkg is None:
            self.label_nobkg.setVisible(True)
            self.label_yesbkg.setVisible(False)
            self.th_type.setVisible(True)
            self.threshold_type_changed(self.th_type.currentText())
        else:
            self.label_nobkg.setVisible(False)
            self.label_yesbkg.setVisible(True)
            self.th_type.setVisible(False)
            self.threshold_type_changed("using_background")

    def setValue(self, value: Sequence[float] | None):
        if value is None:
            value = (0, 130)  # default value
        value = (max(value[0], 0), min(value[1], 255))
        if value[0] == self.simple_slider.min:
            self.simple_slider.set_value(int(value[1]))
            self.th_type.setCurrentText("Dark_animals")
        elif value[1] == self.simple_slider.max:
            self.simple_slider.set_value(int(value[0]))
            self.th_type.setCurrentText("Bright animals")
        else:
            self.range_slider.setValue((int(value[0]), int(value[1])))
            self.th_type.setCurrentText("Custom")

    def value(self) -> tuple[int, int]:
        if self.range_slider.isVisible():
            return self.range_slider.value()
        return (
            (self.simple_slider.value(), self.simple_slider.max)
            if self.simple_slider.inverted
            else (self.simple_slider.min, self.simple_slider.value())
        )

    def setToolTips(self, tooltip_nobkg: str, tooltip_yesbkg: str):
        self.label_nobkg.setToolTip(tooltip_nobkg)
        self.th_type.setToolTip(tooltip_nobkg)
        self.label_yesbkg.setToolTip(tooltip_yesbkg)
        self.range_slider.setToolTip(tooltip_nobkg)
        # self.background_slider.setToolTip(tooltip_yesbkg)
