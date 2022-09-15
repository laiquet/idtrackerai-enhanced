from PyQt6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QHBoxLayout,
)
from PyQt6.QtCore import Qt

# from matplotlib.patches import Polygon
from superqt import QLabeledRangeSlider
import logging


logger = logging.getLogger(__name__)


class TrackingIntervalWidget:
    def __init__(self):
        self.label = QLabel("Tracking interval")
        self.range_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.range_slider.setEnabled(False)
        self.range_slider.setFixedHeight(40)

        self.ranges_checkbutton = QCheckBox("Multiple", enabled=False)

        def multiple_range_change_state(state):
            self.label.setText("Tracking interval" + bool(state) * "s")
            self.range_slider.setVisible(not state)
            self.add_interval.setVisible(state)
            self.ranges_text.setVisible(state)

        self.ranges_checkbutton.stateChanged.connect(
            multiple_range_change_state
        )
        self.ranges_checkbutton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.ranges_text = QTextEdit(visible=False)
        self.ranges_text.setPlaceholderText(
            "Example: [0,1000],[1300,2400],..."
        )
        self.ranges_text.setFixedHeight(28)
        self.add_interval = QPushButton("Add interval", visible=False)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.range_slider)
        self.layout.addWidget(self.ranges_text)
        self.layout.addWidget(self.add_interval)
        self.layout.addWidget(self.ranges_checkbutton)

    def update_ranges(self, start, end):
        self.range_slider.setRange(start, end)
        self.range_slider.setValue((start, end))

    def value(self):
        if self.ranges_checkbutton.isChecked():
            return self.ranges_text.toPlainText()
        else:
            return str(self.range_slider.value())
