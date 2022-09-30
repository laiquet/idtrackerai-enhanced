from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import (
    QProgressBar,
    QHBoxLayout,
    QCheckBox,
    QDialog,
    QLabel,
    QPushButton,
)
from PyQt6.QtGui import QPixmap, QImage
from idtrackerai.animals_detection.segmentation_utils import (
    generate_frame_stack,
    generate_background_from_frame_stack,
)
from confapp import conf


class Thread(QThread):
    def __init__(self, pbar, param_funcs):
        super().__init__()
        self.pbar = pbar
        self.param_funcs = param_funcs

    def run(self):
        video_paths = self.param_funcs["video_paths"]()
        episodes = self.param_funcs["episodes"]()
        ROI_mask = self.param_funcs["ROI_mask"]()
        frame_stack = generate_frame_stack(
            video_paths, episodes, progress_bar=self.pbar
        )
        self.bkg = generate_background_from_frame_stack(frame_stack, ROI_mask)


class ImageDisplay(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Background")

        self.image_lbl = QLabel()
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.image_lbl)

    def show(self, img):
        height, width = img.shape
        pixmap = QPixmap(
            QImage(
                img.tobytes(),
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            )
        )

        # Limit the largest dimension to 800 px
        if height > width:
            pixmap = pixmap.scaledToHeight(800)
        else:
            pixmap = pixmap.scaledToWidth(800)

        self.image_lbl.setPixmap(pixmap)
        self.setFixedSize(pixmap.size().width(), pixmap.size().height())
        super().show()


class background_row(QHBoxLayout):
    def __init__(self, param_funcs):
        super().__init__()
        self.param_funcs = param_funcs
        self.CheckBox = QCheckBox("Background subtraction")
        self.CheckBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.CheckBox.stateChanged.connect(self.btnFunc)
        self.view_bkg = QPushButton("View background", visible=False)

        self.view_bkg.clicked.connect(self.view_bkg_clicked)
        self.pbar = QProgressBar(
            minimum=0,
            maximum=conf.NUMBER_OF_FRAMES_FOR_BACKGROUND - 1,
            visible=False,
        )

        self.image_display = ImageDisplay()

        self.addWidget(self.CheckBox)
        self.addWidget(self.pbar)
        self.addWidget(self.view_bkg)
        self.thread = Thread(self.pbar, self.param_funcs)
        self.thread.finished.connect(self.bkg_thread_finished)

    def view_bkg_clicked(self):
        img = self.get_bkg()
        print(img.dtype, img.shape)
        self.image_display.show((255 * img / img.max()).astype("uint8"))

    def btnFunc(self, checked):
        self.pbar.setVisible(checked)
        if checked:
            self.pbar.setValue(0)
            self.thread.start()
        else:
            if self.thread.isRunning():
                self.thread.quit()
            self.view_bkg.setVisible(False)

    def bkg_thread_finished(self):
        self.pbar.setVisible(False)
        self.view_bkg.setVisible(True)

    def get_bkg(self):
        if self.CheckBox.isChecked():
            if not self.thread.isRunning():
                return self.thread.bkg
        return None
