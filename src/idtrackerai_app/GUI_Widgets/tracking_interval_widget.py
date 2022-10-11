from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal
from idtrackerai_app.GUI_Widgets import my_QLabeleRangeSlider
import ast


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

        self.wrong_input_popup = QMessageBox()
        self.wrong_input_popup.setText("Wrong format")
        self.wrong_input_popup.setIcon(QMessageBox.Warning)
        self.wrong_input_popup.setStandardButtons(QMessageBox.Ok)

    def multiple_text_editingFinished(self):
        error_msg = "Please enter a valid interval format"
        n_frames = self.range_slider.maximum()
        try:
            text = self.multiple_text.text().strip()
            if not text:
                self.multiple_text.clearFocus()
                self.has_changed.emit()
                return
            if text[-1] != ",":
                text += ","

            tracking_intervals = list(ast.literal_eval(text))

            assert tracking_intervals
            assert tracking_intervals[0]

            if len(tracking_intervals) == 1:
                # it is a single interval
                self.range_slider.setValue((tracking_intervals[0]))
                self.multiple_text.clearFocus()
                self.multiple_CheckBox.setChecked(False)
                self.has_changed.emit()
                return

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
            self.wrong_input_popup.setInformativeText(error_msg)
            self.wrong_input_popup.exec()
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
