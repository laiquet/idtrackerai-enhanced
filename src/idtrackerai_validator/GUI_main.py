from pathlib import Path

import numpy as np
from idtrackerai_app.GUI_Widgets import VideoPlayer
from idtrackerai_app.widgets_utils import GUIBase
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QPainter
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from idtrackerai import Blob, ListOfBlobs, Video

from .blob_artist import BlobsArtists


class ValidationGUI(GUIBase):
    def __init__(self, session_path: Path | None = None):
        super().__init__()

        self.setWindowTitle("idTracker.ai | Validation GUI")

        self.video_player = VideoPlayer()
        self.info_widget = QListWidget()
        self.info_widget.setAlternatingRowColors(True)

        right_bar = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_bar)
        right_bar.addWidget(self.info_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.video_player)
        splitter.addWidget(right_widget)
        splitter.setSizes([100, 100])
        self.centralWidget().layout().addWidget(splitter)
        self.centralWidget().setEnabled(False)

        self.selected_fragment: int = -1
        self.video_player.painting_time.connect(self.paint)
        self.frame_number = -1

        open_action = QAction("Open session", self)
        open_action.triggered.connect(
            lambda: self.open_session(
                QFileDialog.getExistingDirectory(
                    self, "Open session directory", ".", QFileDialog.ShowDirsOnly  # type: ignore
                )
            )
        )

        self.menuBar().addAction(open_action)

        file_menu = self.menuBar().addMenu("View")

        self.view_labels = QAction("Labels", self)
        self.view_labels.setShortcut("Alt+L")
        file_menu.addAction(self.view_labels)

        self.view_contours = QAction("Contours", self)
        self.view_contours.setShortcut("Alt+C")
        file_menu.addAction(self.view_contours)

        self.view_centroids = QAction("Centroids", self)
        self.view_centroids.setShortcut("Alt+P")
        file_menu.addAction(self.view_centroids)

        self.view_bboxes = QAction("Bounding boxes", self)
        self.view_bboxes.setShortcut("Alt+B")
        file_menu.addAction(self.view_bboxes)

        self.view_trails = QAction("Trails", self)
        self.view_trails.setShortcut("Alt+T")
        file_menu.addAction(self.view_trails)

        for action in file_menu.actions():
            action.setCheckable(True)
            action.setChecked(True)
            action.changed.connect(self.video_player.update)

        self.video_player.canvas.click_event.connect(self.click_on_canvas)

        if session_path is not None:
            self.open_session(session_path)
        self.center_window()

    def open_session(self, session_path: Path | str):
        if not session_path:
            return
        session_path = Path(session_path)
        self.video = Video.load(session_path)
        self.blobs = ListOfBlobs.load(self.video.blobs_no_gaps_path)
        self.trajectories: np.ndarray = np.load(
            self.video.trajectories_folder / "trajectories_wo_gaps.npy",
            allow_pickle=True,
        ).item()["trajectories"]
        temp = self.trajectories.reshape(-1, self.trajectories.shape[1], 1, 2)
        self.segments = np.concatenate([temp[:-1], temp[1:]], axis=2)
        temp = None
        self.video_player.update_video_paths(
            self.video.video_paths,
            self.video.number_of_frames,
            (self.video.original_width, self.video.original_height),
            self.video.frames_per_second,
        )
        self.centralWidget().setEnabled(True)

        cmap = np.row_stack(
            (
                [1.0, 1.0, 1.0, 1.0],
                get_cmap("gist_rainbow")(
                    np.linspace(0, 1, self.video.number_of_animals)
                ),
            )
        )[:, :-1]
        cmap = (cmap * 255).astype(np.uint8)

        self.blobArtists = BlobsArtists(cmap)
        self.video_player.update()

    def click_on_canvas(self, button: int, xdata: float, ydata: float):
        blob = which_blob_clicked(
            (xdata, ydata), self.blobs.blobs_in_video[self.frame_number]
        )
        selected_fragment = -1 if blob is None else blob.fragment_identifier
        need_to_update = selected_fragment != self.selected_fragment
        self.selected_fragment = selected_fragment
        self.frame_number = -1  # this makes info_widget to update
        if need_to_update:
            self.video_player.update()
        # print(f"Clicked button {button} in ({xdata}, {ydata})")

    def update_right_bar(self, blob: Blob | None):
        self.info_widget.clear()
        if blob is not None:
            self.info_widget.addItems(str(blob).splitlines())
        else:
            self.selected_fragment = -1

    def paint(self, painter: QPainter, frame_number: int, frame: np.ndarray):
        update_info_widget = frame_number != self.frame_number
        self.frame_number = frame_number

        selected_blob = self.blobArtists.set_blobs(
            self.view_contours.isChecked(),
            self.view_centroids.isChecked(),
            self.view_bboxes.isChecked(),
            self.view_labels.isChecked(),
            painter,
            self.blobs.blobs_in_video,
            frame_number,
            self.segments,
            self.selected_fragment,
        )

        if update_info_widget:
            self.update_right_bar(selected_blob)

    def processed_keyPressEvent(self, key: int):
        self.video_player.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.video_player.redirect_keyReleaseEvent(key)


def which_blob_clicked(click_xy: tuple[float, float], blobs: list[Blob]) -> Blob | None:
    for blob in blobs:
        if blob.contains_point(click_xy):
            return blob
