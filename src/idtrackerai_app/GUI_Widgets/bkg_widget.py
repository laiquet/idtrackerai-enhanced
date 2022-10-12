from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QProgressBar,
    QHBoxLayout,
    QCheckBox,
    QDialog,
    QPushButton,
)
from idtrackerai.animals_detection.segmentation_utils import (
    generate_frame_stack,
    generate_background_from_frame_stack,
)
from idtrackerai_app.widgets_utils import MplFigure
from confapp import conf


class Thread(QThread):
    def __init__(self, pbar, param_funcs):
        super().__init__()
        self.pbar = pbar
        self.param_funcs = param_funcs
        self.frame_stack = None
        self.bkg = None

    def run(self):
        video_paths = self.param_funcs["video_paths"]()
        episodes = self.param_funcs["episodes"]()
        ROI_mask = self.param_funcs["ROI_mask"]()
        if self.bkg is None:
            if self.frame_stack is None:
                self.frame_stack = generate_frame_stack(
                    video_paths, episodes, progress_bar=self.pbar
                )
            self.bkg = generate_background_from_frame_stack(
                self.frame_stack, ROI_mask
            )


class ImageDisplay(QDialog, MplFigure):
    def __init__(self):
        super().__init__(adapting_zoom=False)
        self.setWindowTitle("Background")

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.fig.canvas)

        self.im = self.ax.imshow(
            [[]],
            cmap="gray",
            vmax=255,
            vmin=0,
            extent=[0, 1, 1, 0],
            interpolation="none",
            animated=True,
            resample=False,
            snap=False,
        )

    def show(self, img):
        height, width = img.shape

        self.im.set_data(img)
        self.im.set_extent([0, width, height, 0])
        self.x_center = img.shape[1] / 2
        self.y_center = img.shape[0] / 2

        ratio = width / height

        QDialog_size = 500
        if width > height:
            window_width = QDialog_size
            windiw_height = int(QDialog_size / ratio)
        else:
            window_width = int(QDialog_size / ratio)
            windiw_height = QDialog_size
        self.setGeometry(100, 100, window_width, windiw_height)
        self.fit_zoom(width, height, fit_to=(window_width, windiw_height))
        super().exec()


class BkgWidget(QHBoxLayout):
    new_bkg_data = pyqtSignal()

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

    def ROI_has_updated(self):
        self.thread.bkg = None
        self.CheckBox.setChecked(False)

    def tracking_interval_has_changed(self):
        self.thread.bkg = None
        self.thread.frame_stack = None
        self.CheckBox.setChecked(False)

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
            self.new_bkg_data.emit()

    def bkg_thread_finished(self):
        self.pbar.setVisible(False)
        self.view_bkg.setVisible(True)
        self.new_bkg_data.emit()

    def get_bkg(self):
        if self.CheckBox.isChecked():
            if not self.thread.isRunning():
                return self.thread.bkg
        return None
