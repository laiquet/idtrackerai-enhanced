from collections.abc import Sequence

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QMessageBox, QWidget

from idtrackerai.GUI_tools import InvertibleSlider, LabelRangeSlider


class AreaThresholds(QWidget):
    valueChanged = Signal(tuple)

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.upper_limit = QCheckBox("Upper\nlimit")
        self.upper_limit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        area_th_label = QLabel("Blob area\nthresholds")
        area_th_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range_slider = LabelRangeSlider(
            parent=self, min=1, max=60000, block_upper=False
        )
        self.range_slider._min_label.setVisible(False)
        self.range_slider.setVisible(False)
        self.simple_slider = InvertibleSlider(1, 10000)
        self.simple_slider.set_inverted(True)
        self._upper_limit_confirmed = False  # Track if user confirmed
        self.upper_limit.toggled.connect(self.upper_limit_changed)
        layout.addWidget(area_th_label)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.simple_slider)
        layout.addWidget(self.upper_limit)

        self.range_slider.valueChanged.connect(self.valueChanged.emit)
        self.simple_slider.valueChanged.connect(
            lambda val: self.valueChanged.emit((val, float("inf")))
        )

    def upper_limit_changed(self, upper_limit: bool):
        if upper_limit and not self._upper_limit_confirmed:
            reply = QMessageBox.question(
                self,
                "Enable Upper Limit?",
                "<qt>This will enable an upper limit for blob areas.<br>"
                "Large blobs from animal crossings <b>should not</b> be filtered out. "
                "Only blobs that do not correspond to animals should be excluded.<br><br>"
                "Do you want to enable the upper limit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._upper_limit_confirmed = True
            else:
                self.upper_limit.setChecked(False)
                return
        self.range_slider.setVisible(upper_limit)
        self.simple_slider.setVisible(not upper_limit)
        self.valueChanged.emit(self.value())

    def setValue(self, value: Sequence[float] | None):
        if value is None:
            value = (50, float("inf"))  # default value
        low_val = int(value[0])
        self.simple_slider.setValue(low_val)
        if value[1] == float("inf"):
            self.range_slider.setValue((low_val, self.range_slider.maximum()))
            self.upper_limit.setChecked(False)
        else:
            self.range_slider.setValue((low_val, int(value[1])))
            self.upper_limit.setChecked(True)
        self.valueChanged.emit(self.value())

    def value(self) -> tuple[float, float]:
        if self.upper_limit.isChecked():
            return self.range_slider.value()
        return (float(self.simple_slider.value()), float("inf"))
