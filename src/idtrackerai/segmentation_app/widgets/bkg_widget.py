from pathlib import Path
from typing import Sequence

import numpy as np
from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import Qt, QThread, QTimer
from qtpy.QtGui import QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from idtrackerai.base.animals_detection.segmentation import (
    generate_background_from_frame_stack,
    generate_frame_stack,
)
from idtrackerai.GUI_tools import Canvas
from idtrackerai.utils import Episode


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
        if hasattr(self, "background_stat") and new_stat == self.background_stat:
            return
        self.background_stat = new_stat
        self.bkg = None
        if hasattr(self, "video_paths"):
            # when the App in inactive, Stat is set but there is no video_paths yet
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
        self.bkg_stat.addItems(("Median", "Mean", "Max", "Min"))
        self.bkg_stat.setCurrentIndex(-1)
        self.bkg_stat.setEnabled(False)
        self.bkg_stat.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.bkg_stat.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )

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
        layout.addWidget(self.view_bkg)
        self.bkg_stat.currentTextChanged.connect(self.bkg_thread.setStat)
        self.bkg_thread.set_progress_value.connect(self.set_progress_value)
        self.bkg_thread.set_progress_max.connect(self.set_progress_maximum)
        self.bkg_thread.started.connect(self.bkg_thread_started)
        self.bkg_thread.finished.connect(self.bkg_thread_finished)

    def set_new_video_paths(self, video_paths, episodes):
        self.video_paths = video_paths
        self.episodes = episodes
        self.bkg_thread.bkg = None
        self.bkg_thread.frame_stack = None
        if not self.checkBox.isChecked():
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
            self.bkg_thread.start()
        else:
            self.view_bkg.setEnabled(False)
            self.new_bkg_data.emit(None)
        self.bkg_stat.setEnabled(checked)

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
        self.progress_bar.reset()
        if self.bkg_thread.bkg is None:
            self.checkBox.setChecked(False)
        else:
            self.view_bkg.setEnabled(True)
        self.new_bkg_data.emit(self.bkg_thread.bkg)

    def getBkg(self):
        return self.bkg_thread.bkg if self.checkBox.isChecked() else None
