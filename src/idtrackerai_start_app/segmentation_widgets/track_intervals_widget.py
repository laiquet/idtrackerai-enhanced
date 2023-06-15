import ast

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QMessageBox, QWidget

from idtrackerai_GUI_tools import LabelRangeSlider


class TrackingIntervalsWidget(QWidget):
    newValue = Signal(object)

    def __init__(self, parent):
        super().__init__()
        self.checkbox = QCheckBox("Tracking interval")
        self.checkbox.clicked.connect(self.checkbox_clicked)
        self.range_slider = LabelRangeSlider(parent=parent, min=0, max=1)

        self.range_slider.setVisible(False)

        self.multiple_CheckBox = QCheckBox("Multiple")
        self.multiple_CheckBox.setVisible(False)
        self.range_slider.valueChanged.connect(self.emit)
        self.checkbox.clicked.connect(self.emit)

        self.multiple_CheckBox.stateChanged.connect(self.multiple_range_change_state)
        self.multiple_text = QLineEdit()
        self.multiple_text.setVisible(False)
        self.multiple_text.setPlaceholderText("Example: [0,1000],[1300,2400],...")
        self.multiple_text.setFixedHeight(28)
        self.multiple_text.editingFinished.connect(self.load_tracking_intervals)

        layout = QHBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.multiple_text)
        layout.addWidget(self.multiple_CheckBox)
        self.setLayout(layout)

    def setValue(self, value):
        if value is None:
            self.checkbox.setChecked(False)
            return
        self.checkbox.setChecked(True)
        self.load_tracking_intervals(value)
        if self.checkbox.isChecked():
            self.multiple_CheckBox.setVisible(True)
            multiple = self.multiple_CheckBox.isChecked()
            self.multiple_text.setVisible(multiple)
            self.range_slider.setVisible(not multiple)

    def load_tracking_intervals(self, tracking_intervals=None):
        error_msg = "Please enter a valid interval format"
        n_frames = self.range_slider.maximum()

        try:
            if not tracking_intervals:
                text = self.multiple_text.text().strip()
                if not text:
                    self.multiple_text.clearFocus()
                    self.multiple_CheckBox.setChecked(False)
                    self.emit()
                    return

                tracking_intervals = ast.literal_eval(text)

            if not all(isinstance(item, (list, tuple)) for item in tracking_intervals):
                tracking_intervals = [tracking_intervals]

            assert tracking_intervals
            assert all(tracking_intervals)

            if len(tracking_intervals) == 1:
                self.range_slider.setValue((tracking_intervals[0]))
                self.multiple_text.clearFocus()
                self.multiple_CheckBox.setChecked(False)
                self.emit()
                return
            self.multiple_CheckBox.setChecked(True)

            processed_intervals = []
            for start, end in tracking_intervals:
                start = min(max(int(start), 0), n_frames)
                end = min(max(int(end), 0), n_frames)
                if start > end:
                    start, end = end, start
                if end - start:
                    processed_intervals.append([start, end])

            self.multiple_text.setText(str(processed_intervals)[1:-1])
            self.multiple_text.clearFocus()
            self.emit()
        except (ValueError, SyntaxError, AssertionError, TypeError):
            QMessageBox.warning(self, "Tracking intervals error", error_msg)
            self.multiple_text.setFocus()
            self.multiple_text.setText("")

    def multiple_range_change_state(self, state):
        self.checkbox.setText("Tracking interval" + bool(state) * "s")
        self.range_slider.setVisible(not state)
        self.multiple_text.setVisible(state)

    def checkbox_clicked(self, checked):
        self.multiple_CheckBox.setVisible(checked)
        if checked:
            if self.multiple_CheckBox.isChecked():
                self.multiple_text.setVisible(True)
            else:
                self.range_slider.setVisible(True)
        else:
            self.multiple_text.setVisible(False)
            self.range_slider.setVisible(False)

    def reset(self, n_frames):
        self.range_slider.setRange(0, n_frames)
        self.range_slider.setValue((0, n_frames))
        self.checkbox.setChecked(False)

    def emit(self):
        self.newValue.emit(self.value())

    def value(self) -> list[list[int]] | None:
        if not self.checkbox.isChecked():
            return None
        if self.multiple_CheckBox.isChecked() and self.multiple_text.text():
            return list(ast.literal_eval(self.multiple_text.text()))

        val = list(self.range_slider.value())
        if (
            val[0] == self.range_slider.minimum()
            and val[1] == self.range_slider.maximum()
        ):
            return None
        return [val]
