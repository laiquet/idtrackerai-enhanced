from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit
from PyQt6.QtCore import Qt, pyqtSignal
import ast
from idtrackerai_app.widgets_utils import MessageBox, LabelRangeSlider


class TrackingIntervalsWidget(QHBoxLayout):
    has_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.checkbox = QCheckBox("Tracking interval")
        self.checkbox.clicked.connect(self.checkbox_clicked)
        self.range_slider = LabelRangeSlider(
            min=0,
            max=1,
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
            self.load_tracking_intervals
        )

        self.addWidget(self.checkbox)
        self.addWidget(self.range_slider)
        self.addWidget(self.multiple_text)
        self.addWidget(self.multiple_CheckBox)

        self.wrong_input_popup = MessageBox(parent, "Wrong format")

    def setValue(self, value):
        if value:
            self.load_tracking_intervals(value)
            self.checkbox.setChecked(True)
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
                    self.has_changed.emit()
                    return

                tracking_intervals = ast.literal_eval(text)

            if not all(
                [
                    isinstance(item, (list, tuple))
                    for item in tracking_intervals
                ]
            ):
                tracking_intervals = [tracking_intervals]

            assert tracking_intervals
            assert all(tracking_intervals)

            if len(tracking_intervals) == 1:
                # it is a single interval
                self.range_slider.setValue((tracking_intervals[0]))
                self.multiple_text.clearFocus()
                self.multiple_CheckBox.setChecked(False)
                self.has_changed.emit()
                return
            self.multiple_CheckBox.setChecked(True)

            for interval in tracking_intervals:
                print(interval)
                assert len(interval) == 2
                interval[0] = int(interval[0])
                interval[1] = int(interval[1])
                if interval[1] < 0 or interval[0] < 0:
                    error_msg = "Negative tracking intervals!"
                    raise ValueError
                if interval[1] > n_frames or interval[0] > n_frames:
                    error_msg = "Tracking intervals outside the video file!"
                    raise ValueError
                if interval[1] <= interval[0]:
                    error_msg = (
                        "In each interval, start frame has "
                        "to be smaller than the end frame"
                    )
                    raise ValueError

            print(tracking_intervals)
            self.multiple_text.setText(str(tracking_intervals)[1:-1])
            self.multiple_text.clearFocus()
            self.has_changed.emit()
        except (ValueError, SyntaxError, AssertionError, TypeError) as e:
            print(e)
            self.wrong_input_popup.exec(message=error_msg)
            self.multiple_text.setFocus()
            self.multiple_text.setText("")

    def multiple_range_change_state(self, state):
        self.checkbox.setText("Tracking interval" + bool(state) * "s")
        self.range_slider.setVisible(not state)
        # self.add_interval.setVisible(state)
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

    def value(self):
        if not self.checkbox.isChecked():
            return None
        if self.multiple_CheckBox.isChecked():
            return self.multiple_text.text()
        else:
            return [self.range_slider.value()]
