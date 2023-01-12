from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QSlider, QVBoxLayout, QWidget, QSizePolicy
from superqt import QLabeledRangeSlider


class LabelRangeSlider(QLabeledRangeSlider):
    newValue = pyqtSignal(object)

    def __init__(self, min, max, start_end_val=None):
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(min, max)
        if start_end_val:
            self.setValue(start_end_val)
        self.setFixedHeight(40)
        self._max_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._min_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.sliderPressed.connect(lambda: self.newValue.emit(self.value()))
        self._slider.sliderReleased.connect(lambda: self.newValue.emit(self.value()))
        self.editingFinished.connect(lambda: self.newValue.emit(self.value()))
        # self.valueChanged.connect(self.newValue.emit)

    def setValue(self, value) -> None:
        super().setValue(value)
        self.newValue.emit(self.value())

    def value(self) -> list[int]:
        return list(super().value())


class WrappedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self.setMaximumHeight(self.heightForWidth(self.width()))

    def setText(self, text):
        # Add Zero-width space in backslashes for proper word wrapping
        super().setText(text.replace("\\", "\\\u200B"))

    def text(self):
        output = super().text()
        return output.replace("\u200B", "")


class ChangeFontSize(QDialog):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setFixedSize(300, 50)
        self.setLayout(QVBoxLayout())
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.layout().addWidget(self.slider)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setValue(parent.font().pointSize())
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.slider.valueChanged.connect(self.slider_changed)
        self.exec()

    def slider_changed(self, value):
        font = self.parent_widget.font()
        font.setPointSize(value)
        self.parent_widget.setFont(font)
