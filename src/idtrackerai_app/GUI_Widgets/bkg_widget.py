from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QProgressBar,
    QHBoxLayout,
    QCheckBox,
    QDialog,
    QPushButton,
)
from idtrackerai.animals_detection.segmentation import (
    generate_frame_stack,
    generate_background_from_frame_stack,
)
from idtrackerai_app.widgets_utils import MplCanvas
from idtrackerai.utils import conf


class BkgComputationThread(QThread):
    progress_changed = pyqtSignal(int)

    def __init__(self, param_funcs):
        super().__init__()
        self.param_funcs = param_funcs
        self.frame_stack = None
        self.bkg = None
        self.abort = False

    def run(self):
        self.abort = False
        video_paths = self.param_funcs["video_paths"]()
        episodes = self.param_funcs["episodes"]()
        ROI_mask = self.param_funcs["ROI_mask"]()
        if self.bkg is None:
            if self.frame_stack is None:
                self.frame_stack = generate_frame_stack(
                    video_paths,
                    episodes,
                    progress_bar=self.progress_changed,
                    abort=lambda: self.abort,
                )
            if self.abort:
                self.frame_stack = None
                self.abort = False
                return
            self.bkg = generate_background_from_frame_stack(
                self.frame_stack,
                ROI_mask,
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
        super().__init__()
        self.setWindowTitle("Background")
        self.canvas = MplCanvas(parent)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.canvas)

        self.im = self.canvas.ax.imshow(
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
        self.canvas.x_center = img.shape[1] / 2
        self.canvas.y_center = img.shape[0] / 2

        ratio = width / height

        QDialog_size = 500
        if width > height:
            window_width = QDialog_size
            windiw_height = int(QDialog_size / ratio)
        else:
            window_width = int(QDialog_size / ratio)
            windiw_height = QDialog_size
        self.setGeometry(100, 100, window_width, windiw_height)
        self.canvas.fit_zoom(
            width, height, fit_to=(window_width, windiw_height)
        )
        super().exec()


class BkgWidget(QHBoxLayout):
    new_bkg_data = pyqtSignal()

    def __init__(self, parent, param_funcs):
        super().__init__()
        self.param_funcs = param_funcs
        self.CheckBox = QCheckBox("Background subtraction")
        self.CheckBox.stateChanged.connect(self.btnFunc)
        self.view_bkg = QPushButton("View background", visible=False)
        self.view_bkg.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.view_bkg.clicked.connect(self.view_bkg_clicked)
        self.pbar = QProgressBar(
            minimum=0,
            maximum=conf.NUMBER_OF_FRAMES_FOR_BACKGROUND - 1,
            visible=False,
        )

        self.image_display = ImageDisplay(parent)

        self.addWidget(self.CheckBox)
        self.addWidget(self.pbar)
        self.addWidget(self.view_bkg)
        self.bkg_thread = BkgComputationThread(self.param_funcs)
        self.bkg_thread.progress_changed.connect(self.update_ProgressBar)
        self.bkg_thread.finished.connect(self.bkg_thread_finished)

    def update_ProgressBar(self, status):
        self.pbar.setValue(status)

    def partial_reset(self):
        self.bkg_thread.bkg = None
        self.CheckBox.setChecked(False)

    def reset(self):
        self.bkg_thread.bkg = None
        self.bkg_thread.frame_stack = None
        self.CheckBox.setChecked(False)

    def view_bkg_clicked(self):
        img = self.getBkg()
        self.image_display.show((255 * img / img.max()).astype("uint8"))

    def btnFunc(self, checked):
        self.pbar.setVisible(checked)
        if checked:
            self.update_ProgressBar(0)
            self.bkg_thread.start()
        else:
            if self.bkg_thread.isRunning():
                self.bkg_thread.quit()
            self.view_bkg.setVisible(False)
            self.new_bkg_data.emit()

    def bkg_thread_finished(self):
        if self.bkg_thread.bkg is None:
            return
        self.pbar.setVisible(False)
        self.view_bkg.setVisible(True)
        self.new_bkg_data.emit()

    def getBkg(self):
        if self.CheckBox.isChecked():
            if not self.bkg_thread.isRunning():
                return self.bkg_thread.bkg
        return None
