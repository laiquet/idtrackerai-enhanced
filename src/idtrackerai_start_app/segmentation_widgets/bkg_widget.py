import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QProgressDialog,
    QToolButton,
    QWidget,
)

from idtrackerai.animals_detection.segmentation import (
    generate_background_from_frame_stack,
    generate_frame_stack,
)
from idtrackerai.utils import conf
from idtrackerai_GUI_tools import Canvas


class BkgComputationThread(QThread):
    progress_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.frame_stack = None
        self.bkg = None
        self.abort = False
        self.finished.connect(
            lambda: self.progress_changed.emit(conf.NUMBER_OF_FRAMES_FOR_BACKGROUND)
        )

    def set_parameters(self, video_paths, episodes, ROI_mask):
        self.video_paths = video_paths
        self.episodes = episodes
        self.ROI_mask = ROI_mask

    def run(self):
        self.abort = False
        if self.bkg is None:
            if self.frame_stack is None:
                self.frame_stack = generate_frame_stack(
                    self.video_paths,
                    self.episodes,
                    progress_bar=self.progress_changed,
                    abort=lambda: self.abort,
                )
            if self.abort:
                self.frame_stack = None
                self.abort = False
                return
            self.bkg = generate_background_from_frame_stack(
                self.frame_stack,
                self.ROI_mask,
                progress_bar=self.progress_changed,
                abort=lambda: self.abort,
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
            QImage(frame.data, width, height, QImage.Format.Format_Grayscale8)
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
    new_bkg_data = pyqtSignal(object)

    def __init__(self, parent: QWidget):
        super().__init__()
        self.checkBox = QCheckBox("Background subtraction")
        self.checkBox.stateChanged.connect(self.CheckBox_changed)
        self.view_bkg = QToolButton()
        self.view_bkg.setText("View background")
        self.bkg_thread = BkgComputationThread()
        self.view_bkg.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.view_bkg.setVisible(False)
        self.progress_bar = QProgressDialog(
            "Computing background",
            "Cancel",
            0,
            conf.NUMBER_OF_FRAMES_FOR_BACKGROUND,
            parent,
        )
        self.progress_bar.cancel()
        self.progress_bar.setMinimumDuration(1000)
        self.progress_bar.setModal(True)
        self.progress_bar.canceled.connect(self.bkg_thread.quit)
        self.view_bkg.clicked.connect(self.view_bkg_clicked)

        self.image_display = ImageDisplay(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.checkBox)
        layout.addWidget(self.view_bkg)
        self.bkg_thread.progress_changed.connect(self.progress_bar.setValue)
        self.bkg_thread.finished.connect(self.bkg_thread_finished)

    def set_ROI(self, ROI_mask):
        self.ROI_mask = ROI_mask
        self.bkg_thread.bkg = None
        self.checkBox.setChecked(False)

    def set_new_video_paths(self, video_paths, episodes):
        self.video_paths = video_paths
        self.episodes = episodes
        self.bkg_thread.bkg = None
        self.bkg_thread.frame_stack = None
        self.checkBox.setChecked(False)

    def view_bkg_clicked(self):
        img = self.bkg_thread.bkg
        assert img is not None
        self.image_display.show((255 * img / img.max()).astype("uint8"))

    def CheckBox_changed(self, checked):
        if checked:
            if not hasattr(self, "video_paths"):
                self.checkBox.setChecked(False)
                return
            self.bkg_thread.set_parameters(
                self.video_paths, self.episodes, self.ROI_mask
            )
            self.bkg_thread.start()
        else:
            self.view_bkg.setVisible(False)
            self.new_bkg_data.emit(None)

    def bkg_thread_finished(self):
        if self.bkg_thread.bkg is None:
            self.checkBox.setChecked(False)
            self.view_bkg.setVisible(False)
        else:
            self.view_bkg.setVisible(True)
        self.new_bkg_data.emit(self.bkg_thread.bkg)

    def getBkg(self):
        return self.bkg_thread.bkg if self.checkBox.isChecked() else None
