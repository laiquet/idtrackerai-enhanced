from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
)

from idtrackerai.animals_detection.segmentation import (
    generate_background_from_frame_stack,
    generate_frame_stack,
)
from idtrackerai.utils import conf
from idtrackerai_app.widgets_utils import MplCanvas


class BkgComputationThread(QThread):
    progress_changed = pyqtSignal(int)

    def __init__(
        self,
    ):
        super().__init__()
        self.frame_stack = None
        self.bkg = None
        self.abort = False

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
        super().__init__()
        self.setWindowTitle("Background")
        self.canvas = MplCanvas(parent)
        self.canvas.new_drawn.connect(
            lambda: self.im.draw(self.canvas.get_renderer())
        )

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
            window_height = int(QDialog_size / ratio)
        else:
            window_width = int(QDialog_size / ratio)
            window_height = QDialog_size
        self.setGeometry(100, 100, window_width, window_height)
        self.canvas.fit_zoom(
            width, height, fit_to=(window_width, window_height)
        )
        super().exec()


class BkgWidget(QHBoxLayout):
    new_bkg_data = pyqtSignal(object)

    def __init__(
        self,
        parent,
    ):
        super().__init__()
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
        self.bkg_thread = BkgComputationThread()
        self.bkg_thread.progress_changed.connect(self.update_ProgressBar)
        self.bkg_thread.finished.connect(self.bkg_thread_finished)

    def update_ProgressBar(self, status):
        self.pbar.setValue(status)

    def set_ROI(self, ROI_mask):
        self.ROI_mask = ROI_mask
        self.bkg_thread.bkg = None
        self.CheckBox.setChecked(False)

    def set_new_video_paths(self, video_paths, episodes):
        self.video_paths = video_paths
        self.episodes = episodes
        self.bkg_thread.bkg = None
        self.bkg_thread.frame_stack = None
        self.CheckBox.setChecked(False)

    def view_bkg_clicked(self):
        img = self.bkg_thread.bkg
        self.image_display.show((255 * img / img.max()).astype("uint8"))

    def btnFunc(self, checked):
        self.pbar.setVisible(checked)
        if checked:
            self.update_ProgressBar(0)
            self.bkg_thread.set_parameters(
                self.video_paths, self.episodes, self.ROI_mask
            )
            self.bkg_thread.start()
        else:
            if self.bkg_thread.isRunning():
                self.bkg_thread.quit()
            self.view_bkg.setVisible(False)
            self.new_bkg_data.emit(None)

    def bkg_thread_finished(self):
        if self.bkg_thread.bkg is None:
            return
        self.pbar.setVisible(False)
        self.view_bkg.setVisible(True)
        self.new_bkg_data.emit(self.bkg_thread.bkg)
