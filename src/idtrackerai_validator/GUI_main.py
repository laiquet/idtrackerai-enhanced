from pathlib import Path

import numpy as np
from idtrackerai_app.GUI_Widgets import VideoPlayer
from idtrackerai_app.widgets_utils import GUIBase
from matplotlib import colormaps
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Text
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout

from idtrackerai import ListOfBlobs, Video


class ValidationGUI(GUIBase):
    cmap = "gist_rainbow"

    def __init__(self, session_path: Path | None = None):
        super().__init__()

        self.setWindowTitle("idTracker.ai | Validation GUI")

        self.video_player = VideoPlayer()
        self.ax = self.video_player.canvas.ax

        main_layout = QHBoxLayout()
        self.centralWidget().setLayout(main_layout)
        main_layout.addWidget(self.video_player)
        self.centralWidget().setEnabled(False)

        self.label_offset = -30
        self.video_player.blit_event.connect(self.draw)

        open_action = QAction("Open session", self)
        open_action.triggered.connect(
            lambda: self.open_session(
                QFileDialog.getExistingDirectory(
                    self, "Open session directory", ".", QFileDialog.ShowDirsOnly
                )
            )
        )

        self.menuBar().addAction(open_action)

        file_menu = self.menuBar().addMenu("View")

        self.view_labels = QAction("Labels", self)
        file_menu.addAction(self.view_labels)

        self.view_contours = QAction("Contours", self)
        file_menu.addAction(self.view_contours)

        self.view_centroids = QAction("Centroids", self)
        file_menu.addAction(self.view_centroids)

        for action in file_menu.actions():
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(self.video_player.update_player)

        if session_path is not None:
            self.open_session(session_path)
        self.center_window()

    def open_session(self, session_path: Path | str):
        if not session_path:
            return
        session_path = Path(session_path)
        self.video = Video.load(session_path)
        self.blobs = ListOfBlobs.load(self.video.blobs_no_gaps_path)
        self.video_player.update_video_paths(
            self.video.video_paths,
            self.video.number_of_frames,
            (self.video.original_width, self.video.original_height),
            self.video.frames_per_second,
        )
        self.centralWidget().setEnabled(True)

        self.contours: list[Polygon] = [
            self.ax.fill([[0.0, 0.0]], facecolor="None", edgecolor="white")[0]
        ]
        self.centroids: list[Line2D] = [self.ax.plot([], [], ".", color="white")[0]]
        self.labels: list[Text] = [
            self.ax.text(0, 0, "None", color="white", size="x-large")
        ]

        cmap = colormaps[self.cmap]
        for i in range(self.video.number_of_animals):
            color = cmap(i / self.video.number_of_animals)
            self.contours.append(
                self.ax.fill([[0.0, 0.0]], facecolor="None", edgecolor=color)[0]
            )
            self.centroids.append(self.ax.plot([], [], ".", color=color)[0])
            self.labels.append(self.ax.text(0, 0, str(i), color=color, size="x-large"))
        self.video_player.update_player()

    def draw(self, renderer: RendererAgg, frame_number: int, frame: np.ndarray):
        for blob in self.blobs.blobs_in_video[frame_number]:
            assert len(blob.final_identities) == len(
                blob.final_centroids_full_resolution
            )

            for identity, centroid in zip(
                blob.final_identities, blob.final_centroids_full_resolution
            ):
                if identity not in (None, 0):
                    if self.view_centroids.isChecked():
                        self.centroids[identity].set_data(centroid)
                        self.centroids[identity].draw(renderer)
                    if self.view_labels.isChecked():
                        self.labels[identity].set_position(
                            (
                                centroid[0]
                                + self.video_player.canvas.zoom * self.label_offset,
                                centroid[1]
                                + self.video_player.canvas.zoom * self.label_offset,
                            )
                        )
                        self.labels[identity].draw(renderer)

            if (
                len(blob.final_identities) == 1
                and blob.final_identities[0] is not None
                and blob.final_identities[0] > 0
            ):
                identity = blob.final_identities[0]
            else:
                identity = 0

            if self.view_contours.isChecked():
                self.contours[identity].set_xy(blob.contour[:, 0, :])
                self.contours[identity].draw(renderer)

    def processed_keyPressEvent(self, key: int):
        self.video_player.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.video_player.redirect_keyReleaseEvent(key)
