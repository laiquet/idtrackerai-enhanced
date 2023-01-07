from pathlib import Path

import numpy as np
from idtrackerai_app.GUI_Widgets import VideoPlayer
from idtrackerai_app.widgets_utils import GUIBase
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Text
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout

from idtrackerai import ListOfBlobs, Video


class ValidationGUI(GUIBase):
    cmap_name = "gist_rainbow"
    contours: list[Polygon]
    centroids: list[Line2D]
    labels: list[Text]

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
                    self, "Open session directory", ".", QFileDialog.ShowDirsOnly  # type: ignore
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

        self.view_bboxes = QAction("Bounding boxes", self)
        file_menu.addAction(self.view_bboxes)

        for action in file_menu.actions():
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(self.video_player.update_player)

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
        self.video_player.update_video_paths(
            self.video.video_paths,
            self.video.number_of_frames,
            (self.video.original_width, self.video.original_height),
            self.video.frames_per_second,
        )
        self.centralWidget().setEnabled(True)

        self.contours = []
        self.centroids = []
        self.labels = []

        self.cmap = np.row_stack(
            (
                [1.0, 1.0, 1.0, 1.0],
                get_cmap("gist_rainbow")(
                    np.linspace(0, 1, self.video.number_of_animals)
                ),
            )
        )[:, :-1]

        for _ in range(self.video.number_of_animals):
            self.contours.append(self.ax.fill([[0.0, 0.0]], facecolor="None")[0])
            self.centroids.append(self.ax.plot([], [], ".")[0])
            self.labels.append(self.ax.text(0, 0, "", size="x-large"))
        self.video_player.update_player()

    def click_on_canvas(self, button: int, xdata: float, ydata: float):
        print(f"Clicked button {button} in ({xdata}, {ydata})")

    def draw(self, renderer: RendererAgg, frame_number: int):
        centroid_indx = 0
        for blob_indx, blob in enumerate(self.blobs.blobs_in_video[frame_number]):

            color = (
                self.cmap[blob.final_identities[0]]
                if len(blob.final_identities) == 1
                and blob.final_identities[0] is not None
                else self.cmap[0]
            )

            self.contours[blob_indx].set(xy=blob.contour[:, 0, :], edgecolor=color)

            for identity, centroid in zip(
                blob.final_identities, blob.final_centroids_full_resolution
            ):
                color = self.cmap[0] if identity is None else self.cmap[identity]

                self.centroids[centroid_indx].set(data=centroid, color=color)
                self.labels[centroid_indx].set(
                    position=(
                        centroid[0] + self.video_player.canvas.zoom * self.label_offset,
                        centroid[1] + self.video_player.canvas.zoom * self.label_offset,
                    ),
                    color=color,
                    text=str(identity),
                )
                centroid_indx += 1

        if self.view_bboxes.isChecked():
            for i in range(blob_indx + 1):
                ...
                # self.bboxes[i].draw(renderer)

        if self.view_contours.isChecked():
            for i in range(blob_indx + 1):
                self.contours[i].draw(renderer)

        if self.view_centroids.isChecked():
            for i in range(centroid_indx):
                self.centroids[i].draw(renderer)

        if self.view_labels.isChecked():
            for i in range(centroid_indx):
                self.labels[i].draw(renderer)

    def processed_keyPressEvent(self, key: int):
        self.video_player.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.video_player.redirect_keyReleaseEvent(key)
