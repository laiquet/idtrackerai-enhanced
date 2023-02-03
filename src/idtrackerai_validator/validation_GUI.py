from itertools import compress
from pathlib import Path
from typing import Iterable

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from rich.progress import track

from idtrackerai import Blob, ListOfBlobs, Video
from idtrackerai.utils import resolve_path
from idtrackerai_GUI_tools import CustomPainter, GUIBase, VideoPlayer

from .validator_widgets import (
    ErrorsExplorer,
    IdGroups,
    IdLabels,
    SetupPoints,
    find_selected_blob,
    paintBlobs,
    paintTrails,
)

parent_dir = Path(__file__).parent
for file in parent_dir.glob("cmap_*"):
    general_cmap = np.loadtxt(parent_dir / file, dtype=np.uint8)
assert general_cmap is not None

IDTRACKERAI_SHORT_KEYS = {
    "Go to next crossing.": "Ctrl+S",
    "Go to previous crossing.": "Ctrl+A",
    "Check/Uncheck add centroid.": "Ctrl+C",
    "Check/Uncheck add blob.": "Ctrl+B",
    "Delete centroid.": "Ctrl+D",
}
SELECT_POINT_DIST = 100


class SelectId(QDialog):
    def __init__(self, parent: QWidget, n_animals: int):
        super().__init__(parent)
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(n_animals)
        self.setLayout(QVBoxLayout())
        self.description = QLabel()
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setWordWrap(True)

        buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.layout().addWidget(self.description)
        self.layout().addWidget(self.spinbox)
        self.layout().addWidget(buttonBox)

    def exec_with_description(
        self, description: str, default: int | None
    ) -> int | None:
        self.description.setText(description)
        if default is not None:
            self.spinbox.setValue(default)
        self.spinbox.selectAll()
        accepted = super().exec()
        if not accepted:
            return None
        return self.spinbox.value()


class ValidationGUI(GUIBase):
    def __init__(self, session_path: Path | None = None):
        super().__init__()

        self.setWindowTitle("idTracker.ai | Validation GUI")
        self.documentation_url = "https://idtrackerai.readthedocs.io/en/latest/"

        self.video_player = VideoPlayer(self)
        self.following_label = QLabel()
        self.following_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_widget = QListWidget()
        self.info_widget.setAlternatingRowColors(True)

        self.id_groups = IdGroups(self)
        self.id_groups.needToDraw.connect(self.video_player.update)

        self.errorsExplorer = ErrorsExplorer()
        self.errorsExplorer.go_to_error.connect(self.go_to_error)

        self.id_labels = IdLabels()
        self.id_labels.needToDraw.connect(self.video_player.update)

        self.setup_points = SetupPoints(self)

        self.setup_points.needToDraw.connect(self.video_player.update)
        self.video_player.canvas.click_event.connect(self.setup_points.click_event)

        right_bar = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_bar)
        right_bar.addWidget(self.following_label)
        right_bar.addWidget(self.info_widget)
        tabs = QTabWidget()
        tabs.addTab(self.id_groups, "Groups")
        tabs.addTab(self.id_labels, "Labels")
        tabs.addTab(self.setup_points, "Setup Points")
        tabs.currentChanged.connect(self.video_player.update)
        right_bar.addWidget(tabs)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.video_player.layout().setContentsMargins(8, 0, 8, 8)
        splitter.addWidget(self.errorsExplorer)
        splitter.addWidget(self.video_player)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        self.centralWidget().layout().addWidget(splitter)
        self.centralWidget().setEnabled(False)
        self.centralWidget().layout().setContentsMargins(8, 0, 8, 0)

        self.selected_id: int | None = None
        self.selected_blob: Blob | None = None
        self.selection_last_location: Iterable[float] | None = None

        self.video_player.painting_time.connect(self.paint)
        self.frame_number = -1

        session_menu = self.menuBar().addMenu("Session")

        open_action = QAction("Open session", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton)
        )
        open_action.triggered.connect(
            lambda: self.open_session(
                QFileDialog.getExistingDirectory(
                    self, "Open session directory", ".", QFileDialog.ShowDirsOnly  # type: ignore
                )
            )
        )
        session_menu.addAction(open_action)

        save_action = QAction("Save session", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton)
        )
        save_action.triggered.connect(self.save_session)
        session_menu.addAction(save_action)

        drawing_flags = self.menuBar().addMenu("Draw")

        self.view_labels = QAction("Labels", self)
        self.view_labels.setShortcut("Alt+L")
        drawing_flags.addAction(self.view_labels)

        self.view_contours = QAction("Contours", self)
        self.view_contours.setShortcut("Alt+C")
        drawing_flags.addAction(self.view_contours)

        self.view_centroids = QAction("Centroids", self)
        self.view_centroids.setShortcut("Alt+P")
        drawing_flags.addAction(self.view_centroids)

        self.view_bboxes = QAction("Bounding boxes", self)
        self.view_bboxes.setShortcut("Alt+B")
        drawing_flags.addAction(self.view_bboxes)

        self.view_trails = QAction("Trails", self)
        self.view_trails.setShortcut("Alt+T")
        drawing_flags.addAction(self.view_trails)

        for action in drawing_flags.actions():
            action.setCheckable(True)
            action.setChecked(True)
            action.changed.connect(self.video_player.update)

        self.view_labels.setChecked(True)
        self.view_contours.setChecked(True)
        self.view_centroids.setChecked(True)
        self.view_bboxes.setChecked(False)
        self.view_trails.setChecked(True)

        self.video_player.canvas.click_event.connect(self.click_on_canvas)
        self.video_player.canvas.double_click_event.connect(self.double_click_on_canvas)

        self.center_window()
        if session_path is not None:
            QTimer.singleShot(0, lambda: self.open_session(session_path))

    def go_to_error(self, start: int, where: Iterable[float] | None, id: int):
        if where is None:
            where = np.nanmean(self.trajectories[start], axis=0)

        if where is not None:
            # Set the zoom to view ~50 time steps in the current canvas width
            self.video_player.center_canvas_at(
                *where, zoom_scale=50 * self.median_speed
            )
            self.selected_id = id
            self.selection_last_location = where
        self.video_player.setCurrentFrame(start, force_update=True)

    def save_session(self):
        self.video.identities_labels = self.id_labels.get_labels()[1:]
        self.video.identities_groups = self.id_groups.get_groups()
        self.video.setup_points = self.setup_points.get_points()
        self.video.save()
        self.blobs.save(self.video.blobs_path_validated)
        # TODO save trajectories

    def open_session(self, session_path: Path | str):
        if not session_path:
            return
        session_path = resolve_path(session_path)
        self.video = Video.load(session_path)
        blobs_paths_candidates = [
            self.video.blobs_path_validated,
            self.video.blobs_no_gaps_path,
            self.video.blobs_path,
        ]
        found = False
        for path in blobs_paths_candidates:
            if path.is_file():
                self.blobs = ListOfBlobs.load(path)
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"List of blobs not found on any of {blobs_paths_candidates}"
            )

        self.video_player.update_video_paths(
            self.video.video_paths,
            self.video.number_of_frames,
            (self.video.original_width, self.video.original_height),
            self.video.frames_per_second,
            res_reduct=self.video.resolution_reduction,
        )
        self.n_animals = self.video.number_of_animals
        self.n_frames = self.video.number_of_frames
        self.generate_trajectories(self.blobs.blobs_in_video)
        self.median_speed = np.nanmedian(
            np.sqrt(np.sum(np.diff(self.trajectories, axis=0) ** 2, axis=-1))
        )
        self.centralWidget().setEnabled(True)
        self.select_id_dialog = SelectId(self, self.video.number_of_animals)

        cmap = [(255, 255, 255)] + list(
            general_cmap[np.linspace(0, 255, self.video.number_of_animals, dtype=int)]
        )
        self.cmap = [QColor(*color) for color in cmap]
        self.cmap_alpha = [QColor(*color, alpha=77) for color in cmap]

        self.id_groups.load_groups(self.video.identities_groups)
        self.id_labels.load_labels(self.video.identities_labels)
        self.setup_points.load_points(self.video.setup_points)
        self.errorsExplorer.set_references(
            self.trajectories, self.all_identified, self.duplicated
        )
        self.video_player.update()

    def click_on_canvas(self, button: int, zoom: float, x: float, y: float):

        self.selected_blob, self.selected_id, self.selection_last_location = clicked_id(
            self.blobs.blobs_in_video[self.frame_number], x, y
        )

        self.id_groups.selected_id(self.selected_id)
        self.frame_number = -1  # this makes info_widget to update
        self.video_player.update()

    def double_click_on_canvas(self, button: int, zoom: float, x: float, y: float):
        if self.selected_blob is not None and not self.id_groups.editting_name:
            assert self.selection_last_location is not None
            new_id = self.select_id_dialog.exec_with_description(
                "Select the new identity", default=self.selected_id
            )
            if new_id is not None:
                self.selected_blob.update_identity(
                    self.selected_id, new_id, self.selection_last_location
                )
                lower, upper = self.selected_blob.propagate_identity(
                    self.selected_id, new_id, self.selection_last_location
                )
                self.update_trajectories_range(lower, upper)

    def update_right_bar(self, blob: Blob | None):
        self.info_widget.clear()
        if blob is not None:
            self.info_widget.addItems(str(blob).splitlines())
        self.following_label.setText(
            "" if self.selected_id is None else f"Following identity {self.selected_id}"
        )

    def paint(self, painter: CustomPainter, frame_number: int, frame: np.ndarray):

        if self.id_groups.is_active():
            cmap, cmap_alpha = self.id_groups.get_cmaps(self.video.number_of_animals)
        else:
            cmap, cmap_alpha = self.cmap, self.cmap_alpha

        update_info_widget = frame_number != self.frame_number
        self.frame_number = frame_number

        self.selected_blob, self.selection_last_location = find_selected_blob(
            self.blobs.blobs_in_video[self.frame_number],
            self.selected_id,
            self.selection_last_location,
        )

        if self.view_trails.isChecked():
            paintTrails(self.frame_number, painter, self.trajectories, cmap)

        paintBlobs(
            self.view_contours.isChecked(),
            self.view_centroids.isChecked(),
            self.view_bboxes.isChecked(),
            self.view_labels.isChecked(),
            painter,
            self.blobs.blobs_in_video[self.frame_number],
            cmap,
            cmap_alpha,
            self.selected_blob,
            self.selection_last_location,
            self.id_labels.get_labels(),
        )

        if self.setup_points.isVisible():
            self.setup_points.paint_on_canvas(painter)

        if update_info_widget:
            self.update_right_bar(self.selected_blob)

    def processed_keyPressEvent(self, key: int):
        if key in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.id_groups.enter_pressed()
            self.setup_points.enter_pressed()
        self.video_player.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.video_player.redirect_keyReleaseEvent(key)

    def update_trajectories_range(self, start: int, finish: int):
        # Update trajectories on errorsExplorer
        finish += 1
        self.trajectories[start:finish] = np.nan
        for frame_number, blobs_in_frame in enumerate(
            self.blobs.blobs_in_video[start:finish], start
        ):
            for blob in blobs_in_frame:
                for identity, centroid in zip(
                    blob.final_identities, blob.final_centroids
                ):
                    if identity not in (None, 0):
                        self.trajectories[frame_number, identity - 1] = centroid

    def update_trajectories(self):
        for frame_number in compress(range(self.n_frames), self.frames_to_update):
            self.trajectories[frame_number] = np.nan
            for blob in self.blobs.blobs_in_video[frame_number]:
                for identity, centroid in zip(
                    blob.final_identities, blob.final_centroids
                ):
                    if identity not in (None, 0):
                        self.trajectories[blob.frame_number, identity - 1] = centroid

    def generate_trajectories(self, blobs_in_video: list[list[Blob]]):
        number_of_frames = len(blobs_in_video)
        self.trajectories = np.full((number_of_frames, self.n_animals, 2), np.NaN)
        self.all_identified = np.ones((number_of_frames), bool)
        self.duplicated = np.zeros((number_of_frames, self.n_animals), bool)
        self.frames_to_update = np.zeros(number_of_frames, bool)
        ids_in_frame: set[int] = set()
        for blobs_in_frame in track(
            blobs_in_video, description="Analyzing trajectories"
        ):
            ids_in_frame.clear()
            for blob in blobs_in_frame:
                for identity, centroid in zip(
                    blob.final_identities, blob.final_centroids
                ):

                    if identity not in (None, 0):
                        self.trajectories[blob.frame_number, identity - 1] = centroid
                        if identity in ids_in_frame:
                            self.duplicated[blob.frame_number, identity - 1] = True
                        ids_in_frame.add(identity)
                    else:
                        self.all_identified[blob.frame_number] = False


def clicked_id(
    blobs: list[Blob], x, y
) -> tuple[Blob, int | None, tuple[float, float]] | tuple[None, int, None]:
    distances_to_centroids: list[
        tuple[Blob, int | None, tuple[float, float], float]
    ] = []

    for blob in blobs:
        if blob.contains_point((x, y)):
            for id, centroid in zip(blob.final_identities, blob.final_centroids):
                dist = (centroid[0] - x) ** 2 + (centroid[1] - y) ** 2
                distances_to_centroids.append((blob, id, centroid, dist))
            break

    if distances_to_centroids:
        return sorted(distances_to_centroids, key=lambda x: x[-1])[0][:-1]

    for blob in blobs:
        for id, centroid in zip(blob.final_identities, blob.final_centroids):
            dist = (centroid[0] - x) ** 2 + (centroid[1] - y) ** 2
            if dist < SELECT_POINT_DIST:
                distances_to_centroids.append((blob, id, centroid, dist))

    if distances_to_centroids:
        return sorted(distances_to_centroids, key=lambda x: x[-1])[0][:-1]

    return None, -1, None
