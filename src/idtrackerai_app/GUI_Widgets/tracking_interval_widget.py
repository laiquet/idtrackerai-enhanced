from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
)
from PyQt6.QtCore import Qt

# from matplotlib.patches import Polygon
from superqt import QLabeledRangeSlider, QLabeledDoubleRangeSlider


class TrackingIntervalWidget:
    def __init__(self):

        self.checkbox = QCheckBox("Tracking interval")
        self.checkbox.clicked.connect(self.checkbox_clicked)
        self.range_slider = QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal
        )
        self.range_slider.setVisible(False)
        self.range_slider.setFixedHeight(40)

        self.multiple_CheckBox = QCheckBox("Multiple", visible=False)

        def multiple_range_change_state(state):
            self.checkbox.setText("Tracking interval" + bool(state) * "s")
            self.range_slider.setVisible(not state)
            # self.add_interval.setVisible(state)
            self.multiple_text.setVisible(state)

        self.multiple_CheckBox.stateChanged.connect(
            multiple_range_change_state
        )
        self.multiple_CheckBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.multiple_text = QLineEdit(visible=False)
        self.multiple_text.setPlaceholderText(
            "Example: [0,1000],[1300,2400],..."
        )
        self.multiple_text.setFixedHeight(28)
        # self.add_interval = QPushButton("Add interval", visible=False)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.checkbox)
        self.layout.addWidget(self.range_slider)
        self.layout.addWidget(self.multiple_text)
        # self.layout.addWidget(self.add_interval)
        self.layout.addWidget(self.multiple_CheckBox)

    def checkbox_clicked(self, checked):
        self.multiple_CheckBox.setVisible(checked)
        if checked:
            if self.multiple_CheckBox.isChecked():
                self.multiple_text.visible = True
            else:
                self.range_slider.setVisible(True)
        else:
            self.multiple_text.visible = False
            self.range_slider.setVisible(False)

    def update_ranges(self, start, end):
        self.range_slider.setRange(start, end)
        self.range_slider.setValue((start, end))

    def value(self):
        if not self.checkbox.isChecked():
            return None
        if self.multiple_CheckBox.isChecked():
            return self.ranges_text.toPlainText()
        else:
            return [self.range_slider.value()]
