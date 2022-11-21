from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QSpinBox,
    QLineEdit,
)
from matplotlib.pyplot import rcParams

# from idtrackerai.utils import conf
# from PyQt6.QtCore import Qt, QCoreApplication
# from matplotlib.backend_bases import KeyEvent as matplotlib_KeyEvent
# from PyQt6.QtGui import QKeyEvent as PyQt_KeyEvent
# from pathlib import Path
# from idtrackerai_app.GUI_Widgets import (
#     VideoPlayerWidget,
#     ROIWidget,
#     SetupPointsWidget,
#     OpenVideoWidget,
#     BkgWidget,
#     TrackingIntervalsWidget,
#     BlobInfoWidget,
# )
from idtrackerai_app.widgets_utils import LabelRangeSlider
import logging
import json
import numpy as np
from pathlib import Path
from idtrackerai_app.widgets_utils import VideoPathHolder_Cls, VideoPlayer
from idtrackerai import Video, ListOfBlobs
from matplotlib import colormaps as all_cmaps


class Window(QWidget):
    cmap = all_cmaps["gist_rainbow"]

    def __init__(self):

        logging.debug("Initializing Validator")
        super().__init__()

        # Clean all the default keyboard shortcuts of matplotlib
        for action, keybindings in rcParams.items():
            if action.startswith("keymap."):
                keybindings.clear()

        self.setWindowTitle("idTracker.ai | Validation GUI")
        self.setGeometry(100, 60, 1000, 800)

        self.setLayout(QHBoxLayout())
        self.video_player = VideoPlayer()

        session_path = Path("/home/jordi/idtrackerai/session_test_old")
        # self.video = Video.load(session_path / "video_object.npy")

        # blobs_path = Path(self.video.blobs_no_gaps_path)
        # if not blobs_path.exists():
        #     blobs_path = self.video.blobs_path

        # self.blobs = ListOfBlobs.load(blobs_path)
        self.blobs = ListOfBlobs.load(
            session_path / "preprocessing" / "blobs_collection.npy"
        )

        self.video_player.update_video_paths(
            ["/home/jordi/idtrackerai/light_video.avi"],
            508,
            (1160, 938),
            25,
        )
        self.ax = self.video_player.canvas.ax
        self.n_animals = 8
        self.layout().addWidget(self.video_player)
        self.video_player.frame_ready_to_draw.connect(self.draw_patches)

        self.drawned = []
        self.contours = []
        self.centroids = []
        self.labels = []

        for i in range(self.n_animals):
            color = self.cmap(i / self.n_animals)
            self.contours.append(self.ax.plot([], [], color=color)[0])
            self.centroids.append(self.ax.plot([], [], ".", color=color)[0])
            self.labels.append(
                self.ax.text(0, 0, str(i), color=color, size="x-large")
            )
        self.label_offset = np.asarray([-30, -30])
        self.draw_patches(0)

    def draw_patches(self, frame):
        # blobs = [None] * self.n_animals

        # for blob in self.blobs.blobs_in_video[frame]:
        #     if blob.identity:
        #         blobs[blob.identity - 1] = (
        #             blob.centroid,
        #             blob.contour[:, 0, :].T + 0.5,
        #         )

        # for i, blob in enumerate(blobs):
        #     if blob:
        #         centroid, contour = blob
        #         self.labels[i].set_position(centroid + self.label_offset)
        #         self.contours[i].set_data(*contour)
        #         self.centroids[i].set_data(*centroid)
        for i, blob in enumerate(self.blobs.blobs_in_video[frame]):
            self.contours[i].set_data(*blob.contour.T)
        i += 1
        for j in range(i, self.n_animals):
            self.contours[j].set_data([], [])
