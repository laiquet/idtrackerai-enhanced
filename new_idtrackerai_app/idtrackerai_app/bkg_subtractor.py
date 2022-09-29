from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import (
    QProgressBar,
    QHBoxLayout,
    QCheckBox,
)
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


class background_row(QHBoxLayout):
    def __init__(self, param_funcs):
        super().__init__()
        self.param_funcs = param_funcs
        self.CheckBox = QCheckBox("Subtract background")
        self.CheckBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.CheckBox.stateChanged.connect(self.btnFunc)

        self.pbar = QProgressBar(
            minimum=0,
            maximum=conf.NUMBER_OF_FRAMES_FOR_BACKGROUND - 1,
            visible=False,
        )

        self.addWidget(self.CheckBox)
        self.addWidget(self.pbar)
        self.thread = Thread(self.pbar, self.param_funcs)
        self.thread.finished.connect(self.bkg_thread_finished)

    def btnFunc(self, checked):
        self.pbar.setVisible(checked)
        if checked:
            self.pbar.setValue(0)
            self.thread.start()
        else:
            if self.thread.isRunning():
                self.thread.quit()

    def bkg_thread_finished(self):
        self.pbar.setVisible(False)
        self.CheckBox.setEnabled(True)

    def get_bkg(self):
        if self.CheckBox.isChecked():
            if not self.thread.isRunning():
                return self.thread.bkg
        return None
