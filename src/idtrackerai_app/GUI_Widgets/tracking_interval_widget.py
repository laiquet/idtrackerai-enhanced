from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal
from idtrackerai_app.GUI_Widgets import my_QLabeleRangeSlider


class TrackingIntervalWidget(QHBoxLayout):
    has_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.checkbox = QCheckBox("Tracking interval")
        self.checkbox.clicked.connect(self.checkbox_clicked)
        self.range_slider = my_QLabeleRangeSlider(
            min=0,
            max=1,
            start_val=0,
            end_val=1,
        )

        self.range_slider.setVisible(False)
        self.range_slider.setFixedHeight(40)

        self.multiple_CheckBox = QCheckBox("Multiple", visible=False)
        self.range_slider.has_changed.connect(self.has_changed.emit)
        self.checkbox.clicked.connect(self.has_changed.emit)

        self.multiple_CheckBox.stateChanged.connect(
            self.multiple_range_change_state
        )
        self.multiple_CheckBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.multiple_text = QLineEdit(visible=False)
        self.multiple_text.setPlaceholderText(
            "Example: [0,1000],[1300,2400],..."
        )
        self.multiple_text.setFixedHeight(28)
        self.multiple_text.editingFinished.connect(
            self.multiple_text_editingFinished
        )

        self.addWidget(self.checkbox)
        self.addWidget(self.range_slider)
        self.addWidget(self.multiple_text)
        self.addWidget(self.multiple_CheckBox)

    def multiple_text_editingFinished(self):
        print("finish")
        # TODO Validate the input given by user (check limits, format...)
        self.multiple_text.clearFocus()
        self.has_changed.emit()

    def multiple_range_change_state(self, state):
        self.checkbox.setText("Tracking interval" + bool(state) * "s")
        self.range_slider.setVisible(not state)
        # self.add_interval.setVisible(state)
        self.multiple_text.setVisible(state)

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
            return self.multiple_text.text()
        else:
            return [self.range_slider.value()]
