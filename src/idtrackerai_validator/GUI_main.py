from pathlib import Path

import numpy as np
from idtrackerai_app.GUI_Widgets import VideoPlayer
from idtrackerai_app.widgets_utils import GUIBase
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QListWidget, QVBoxLayout

from idtrackerai import Blob, ListOfBlobs, Video

from .blob_artist import BlobsArtists


class ValidationGUI(GUIBase):
    def __init__(self, session_path: Path | None = None):
        super().__init__()

        self.setWindowTitle("idTracker.ai | Validation GUI")

        self.video_player = VideoPlayer()
        self.info_widget = QListWidget()
        self.info_widget.setAlternatingRowColors(True)
        self.ax = self.video_player.canvas.ax

        main_layout = QHBoxLayout()
        right_bar = QVBoxLayout()
        right_bar.addWidget(self.info_widget)
        self.centralWidget().setLayout(main_layout)
        main_layout.addWidget(self.video_player)
        main_layout.addLayout(right_bar)
        self.centralWidget().setEnabled(False)

        self.selected_fragment: int = -1
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

        for action in file_menu.actions():
            action.setCheckable(True)
            action.setChecked(True)
            action.changed.connect(self.video_player.update_player)

        self.video_player.canvas.click_event.connect(self.click_on_canvas)
        self.video_player.canvas.artist_clicked.connect(self.artist_clicked)

        if session_path is not None:
            self.open_session(session_path)
        self.center_window()

    def artist_clicked(self, artist):

        """
        'guiEvent.xdata': 731.2025709924056
        'guiEvent.ydata': 379.3035595807762
        'guiEvent.button': <MouseButton.LEFT: 1>
        'guiEvent.key': None
        'guiEvent.step': 0
        'guiEvent.dblclick': False
        'artist': <matplotlib.patches.Polygon object at 0x7f4b13774310>"""

        selected_fragment = (
            -1 if artist is None else artist.associated_blob.fragment_identifier
        )
        need_to_update = selected_fragment != self.selected_fragment
        self.selected_fragment = selected_fragment
        if need_to_update:
            self.video_player.update_player()
        # print(event.mouseevent.__dict__)

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

        cmap = np.row_stack(
            (
                [1.0, 1.0, 1.0, 1.0],
                get_cmap("gist_rainbow")(
                    np.linspace(0, 1, self.video.number_of_animals)
                ),
            )
        )[:, :-1]

        self.blobArtists = BlobsArtists(self.video.number_of_animals, self.ax, cmap)
        self.video_player.update_player()

    def click_on_canvas(self, button: int, xdata: float, ydata: float):
        pass
        # print(f"Clicked button {button} in ({xdata}, {ydata})")

    def draw(self, renderer: RendererAgg, frame_number: int):

        selected_blob = self.blobArtists.set_blobs(
            self.blobs.blobs_in_video[frame_number], self.selected_fragment
        )

        self.info_widget.clear()
        if selected_blob is not None:
            self.info_widget.addItems(str(selected_blob).splitlines())
        else:
            self.selected_fragment = -1

        if self.view_bboxes.isChecked():
            self.blobArtists.draw_bboxes(renderer)

        if self.view_contours.isChecked():
            self.blobArtists.draw_contours(renderer)

        if self.view_centroids.isChecked():
            self.blobArtists.draw_centroids(renderer)

        if self.view_labels.isChecked():
            self.blobArtists.draw_labels(renderer)

    def processed_keyPressEvent(self, key: int):
        self.video_player.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.video_player.redirect_keyReleaseEvent(key)
