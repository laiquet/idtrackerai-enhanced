from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import QDialog, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget
from superqt import QLabeledRangeSlider, QLabeledSlider


class LabeledSlider(QLabeledSlider):
    def __init__(self, parent: QWidget, min, max):
        self.parent_widget = parent
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(min, max)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        self._label.wheelEvent = lambda event: event.accept()

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            style = (
                f"color: #{self.palette().text().color().rgba():x}"
                ";background:transparent; border: 0;"
            )
            self._label.setStyleSheet(style)
            self._slider.setPalette(self.parent_widget.palette())

        elif event.type() == QEvent.Type.FontChange:
            style = (
                f"color: #{self.palette().text().color().rgba():x}; background:"
                f"transparent; border: 0; font-size:{self.font().pointSize()}px"
            )
            self._label.setStyleSheet(style)
            self._label._update_size()


class LabelRangeSlider(QLabeledRangeSlider):
    def __init__(self, parent: QWidget, min, max, start_end_val=None, block_upper=True):
        self.parent_widget = parent
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(min, max)
        if start_end_val:
            self.setValue(start_end_val)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        self._min_label.wheelEvent = lambda event: event.ignore()
        self._max_label.wheelEvent = lambda event: event.ignore()
        for handle in self._handle_labels:
            handle.wheelEvent = lambda event: event.ignore()

        self._min_label.setReadOnly(True)
        if block_upper:
            self._max_label.setReadOnly(True)

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            style = (
                f"color: #{self.palette().text().color().rgba():x}"
                ";background:transparent; border: 0;"
            )
            self._max_label.setStyleSheet(style)
            self._min_label.setStyleSheet(style)
            for handle in self._handle_labels:
                handle.setStyleSheet(style)
            self._slider.setPalette(self.parent_widget.palette())

        elif event.type() == QEvent.Type.FontChange:
            style = (
                f"color: #{self.palette().text().color().rgba():x}; background:"
                f"transparent; border: 0; font-size:{self.font().pointSize()}px"
            )
            self._min_label.setStyleSheet(style)
            self._max_label.setStyleSheet(style)
            self._max_label._update_size()
            self._min_label._update_size()
            for handle in self._handle_labels:
                handle.setStyleSheet(style)
                handle._update_size()


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
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.slider.valueChanged.connect(self.slider_changed)
        self.exec()

    def slider_changed(self, value):
        font = self.parent_widget.font()
        font.setPointSize(value)
        self.parent_widget.setFont(font)
