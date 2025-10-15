from collections.abc import Sequence
from pathlib import Path

import numpy as np
from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import Qt, QThread, QTimer
from qtpy.QtGui import QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from idtrackerai.base.animals_detection import (
    generate_background_from_frame_stack,
    generate_frame_stack,
    load_custom_background,
)
from idtrackerai.GUI_tools import Canvas, WrappedLabel
from idtrackerai.utils import Episode, IdtrackeraiError


def is_standard(stat: str) -> bool:
    return stat.lower() in ("median", "mean", "max", "min")


class BkgComputationThread(QThread):
    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    background_stat: str
    n_frames_for_background: int
    video_paths: Sequence[str | Path]
    episodes: list[Episode]

    def __init__(self):
        super().__init__()
        self.frame_stack = None
        self.bkg = None
        self.abort = False

    def setStat(self, stat: str):
        new_stat = stat.lower()
        self.background_stat = new_stat
        self.bkg = None
        if hasattr(self, "video_paths"):
            # when the App in inactive, Stat is set but there is no video_paths yet
            if not is_standard(self.background_stat):
                # we load the custom background in the main thread because it can raise IdtrackeraiError
                self.bkg = load_custom_background(
                    self.background_stat, self.video_paths[0]
                )
                self.finished.emit()
            else:
                self.start()

    def set_parameters(
        self, video_paths: Sequence[str | Path], episodes: list[Episode]
    ):
        self.video_paths = video_paths
        self.episodes = episodes

    def run(self):
        self.abort = False
        if self.bkg is not None:
            return
        self.set_progress_max.emit(self.n_frames_for_background)
        if self.frame_stack is None:
            self.frame_stack = generate_frame_stack(
                self.episodes,
                self.n_frames_for_background,
                self.set_progress_value,
                lambda: self.abort,
            )
        if self.abort:
            self.frame_stack = None
            self.abort = False
            return

        self.set_progress_value.emit(0)
        self.set_progress_max.emit(0)
        self.bkg = generate_background_from_frame_stack(
            self.frame_stack, self.background_stat
        )

        if self.abort:
            self.frame_stack = None
            self.bkg = None
            self.abort = False
            return

    def quit(self):
        self.abort = True


class ImageDisplay(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Background")
        self.canvas = Canvas()
        self.canvas.painting_time.connect(self.paint_image)

        self.setLayout(QHBoxLayout())
        self.setMinimumSize(200, 50)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.canvas)

    def paint_image(self, painter: QPainter):
        painter.drawPixmap(0, 0, self.pixmap)

    def show(self, frame: np.ndarray):
        height, width = frame.shape
        self.pixmap = QPixmap.fromImage(
            QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8)
        )

        self.canvas.centerX = int(width / 2)
        self.canvas.centerY = int(height / 2)

        ratio = width / height

        QDialog_size = 600
        if width > height:
            window_width = QDialog_size
            window_height = int(QDialog_size / ratio)
        else:
            window_width = int(QDialog_size / ratio)
            window_height = QDialog_size
        self.setGeometry(0, 0, window_width, window_height)
        QTimer.singleShot(0, lambda: self.canvas.adjust_zoom_to(width, height))
        super().exec()


class BkgWidget(QWidget):
    new_bkg_data = Signal(object)  # np.ndarray | None

    def __init__(self):
        super().__init__()
        self.checkBox = QCheckBox("Background\nsubtraction")
        self.checkBox.stateChanged.connect(self.CheckBox_changed)

        self.bkg_stat = QComboBox()
        self.bkg_stat.addItems(("Custom", "Median", "Mean", "Max", "Min"))
        self.bkg_stat.setCurrentIndex(-1)
        self.bkg_stat.setEnabled(False)
        self.bkg_stat.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.bkg_stat.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )
        self.last_stat = "Median"

        self.uploaded_path = WrappedLabel(framed=True)
        self.uploaded_path.clicked.connect(lambda: self.bkg_stat_changed("Custom"))
        self.uploaded_path.setVisible(False)

        self.view_bkg = QToolButton()
        self.view_bkg.setText("View background")
        self.bkg_thread = BkgComputationThread()
        self.view_bkg.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.view_bkg.setEnabled(False)
        self.view_bkg.clicked.connect(self.view_bkg_clicked)

        self.image_display = ImageDisplay(self)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.checkBox)
        layout.addWidget(self.bkg_stat)
        layout.addWidget(self.uploaded_path)
        layout.addWidget(self.view_bkg)
        self.bkg_stat.currentTextChanged.connect(self.bkg_stat_changed)
        self.bkg_thread.set_progress_value.connect(self.set_progress_value)
        self.bkg_thread.set_progress_max.connect(self.set_progress_maximum)
        self.bkg_thread.started.connect(self.bkg_thread_started)
        self.bkg_thread.finished.connect(self.bkg_thread_finished)

    def bkg_stat_changed(self, text: str) -> None:
        if is_standard(text):
            self.bkg_thread.setStat(text)
            return

        assert text.lower() == "custom"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select background image",
            self.uploaded_path.text(),
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff);;All files (*)",
        )
        if not file_path:
            self.set_bkg_stat(self.last_stat)
            return

        self.set_custom_path(file_path)

    def set_custom_path(self, file_path: str):
        try:
            self.bkg_thread.setStat(file_path)
        except IdtrackeraiError as err:
            QMessageBox.critical(None, "Error", str(err))
            self.set_bkg_stat(self.last_stat)
            return

        self.uploaded_path.setText(file_path)
        self.bkg_stat.blockSignals(True)
        self.bkg_stat.setCurrentText("Custom")
        self.bkg_stat.blockSignals(False)

    def set_bkg_stat(self, stat: str):
        if stat.lower() in ("median", "mean", "max", "min", "custom"):
            self.bkg_stat.setCurrentText(stat.capitalize())
        else:
            self.set_custom_path(stat)

    def set_new_video_paths(self, video_paths, episodes):
        self.video_paths = video_paths
        self.episodes = episodes
        self.bkg_thread.bkg = None
        self.bkg_thread.frame_stack = None
        if not self.checkBox.isChecked() or self.bkg_stat.currentText() == "Custom":
            return
        QMessageBox.information(
            self,
            "Background deactivated",
            "The subtracted background depends on the specified video paths. Check"
            " again the background subtraction if desired when finish editing the"
            " video paths.",
        )
        self.checkBox.setChecked(False)

    def view_bkg_clicked(self):
        if self.bkg_thread.bkg is not None:
            self.image_display.show(self.bkg_thread.bkg)

    def CheckBox_changed(self, checked):
        if checked:
            if not hasattr(self, "video_paths"):
                self.checkBox.setChecked(False)
                return
            self.bkg_thread.set_parameters(self.video_paths, self.episodes)
            self.bkg_thread.setStat(
                self.bkg_stat.currentText()
                if is_standard(self.bkg_stat.currentText())
                else self.uploaded_path.text()
            )
        else:
            self.view_bkg.setEnabled(False)
            self.new_bkg_data.emit(None)
        self.bkg_stat.setEnabled(checked)
        self.uploaded_path.setEnabled(checked)

    def bkg_thread_started(self):
        self.progress_bar = QProgressDialog("Computing background", "Cancel", 0, 100)
        self.progress_bar.setMinimumDuration(200)
        self.progress_bar.setModal(True)
        self.progress_bar.setAutoReset(False)
        self.progress_bar.canceled.connect(self.bkg_thread.quit)

    def set_progress_value(self, value: int):
        self.progress_bar.setValue(value)

    def set_progress_maximum(self, value: int):
        self.progress_bar.setMaximum(value)

    def bkg_thread_finished(self):
        if hasattr(self, "progress_bar"):
            self.progress_bar.reset()
        if self.bkg_thread.bkg is None:
            self.checkBox.setChecked(False)
        else:
            self.view_bkg.setEnabled(True)
            self.last_stat = (
                self.bkg_stat.currentText()
                if is_standard(self.bkg_stat.currentText())
                else self.uploaded_path.text()
            )
        self.uploaded_path.setVisible(not is_standard(self.last_stat))
        self.new_bkg_data.emit(self.bkg_thread.bkg)

    def getBkg(self):
        return self.bkg_thread.bkg if self.checkBox.isChecked() else None

    def get_bkg_stat(self) -> None | str:
        if not self.checkBox.isChecked():
            return None
        if self.bkg_stat.currentText() == "Custom":
            return self.uploaded_path.text()
        return self.bkg_stat.currentText().lower()
