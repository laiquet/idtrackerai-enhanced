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

# from confapp import conf
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
from pathlib import Path
from idtrackerai_app.widgets_utils import VideoPathHolder_Cls, VideoPlayer
from idtrackerai import Video


class Window(QWidget):
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
        video_player = VideoPlayer()

        session_path = Path("/home/jordi/idtrackerai/session_test")

        blobs_path = (
            session_path / "preprocessing" / "blobs_collection_no_gaps.npy"
        )
        if not blobs_path.exists():
            blobs_path = (
                session_path / "preprocessing" / "blobs_collection.npy"
            )
        vidobj_path = session_path / "video_object.npy"

        video = Video.load(vidobj_path)
        video_player.update_video_paths(
            video.video_paths,
            video.number_of_frames,
            (video.original_width, video.original_height),
            video.frames_per_second,
        )
        self.layout().addWidget(video_player)
        # blobs = np.load(blobs_path, allow_pickle=True).item()

        # resolution = video.resolution_reduction
