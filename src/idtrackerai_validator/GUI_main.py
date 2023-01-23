from pathlib import Path

import numpy as np
from idtrackerai_app.GUI_Widgets import VideoPlayer
from idtrackerai_app.widgets_utils import CustomQPainter, GUIBase, ListLayout
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
    QVBoxLayout,
    QWidget,
)

from idtrackerai import Blob, ListOfBlobs, Video

from .validator_widgets_and_utils import paintBlobs, IdGroups

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
SELECT_POINT_DIST = 10


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

    def exec_with_description(self, description: str, default: int) -> int | None:
        self.description.setText(description)
        self.spinbox.setValue(default)
        accepted = super().exec()
        if not accepted:
            return None
        return self.spinbox.value()


class ValidationGUI(GUIBase):
    def __init__(self, session_path: Path | None = None):
        super().__init__()

        self.setWindowTitle("idTracker.ai | Validation GUI")

        self.video_player = VideoPlayer()
        self.following_label = QLabel()
        self.following_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_widget = QListWidget()
        self.info_widget.setAlternatingRowColors(True)

        right_bar = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_bar)
        right_bar.addWidget(self.following_label)
        right_bar.addWidget(self.info_widget)
        self.id_groups = IdGroups(self)
        self.id_groups.needToDraw.connect(self.video_player.update)
        right_bar.addWidget(self.id_groups)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.video_player)
        splitter.addWidget(right_widget)
        splitter.setSizes([100, 100])
        self.centralWidget().layout().addWidget(splitter)
        self.centralWidget().setEnabled(False)
        self.centralWidget().layout().setContentsMargins(0, 0, 8, 0)

        self.selected_fragment: int = -1
        self.selected_id: int = -1
        self.selected_blob: Blob | None = None
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
        self.view_trails.setChecked(False)

        self.video_player.canvas.click_event.connect(self.click_on_canvas)
        self.video_player.canvas.double_click_event.connect(self.double_click_on_canvas)

        self.center_window()
        if session_path is not None:
            QTimer.singleShot(0, lambda: self.open_session(session_path))

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
            res_reduct=self.video.resolution_reduction,
        )
        self.centralWidget().setEnabled(True)
        self.select_id_dialog = SelectId(self, self.video.number_of_animals)

        cmap = [(255, 255, 255)] + list(
            general_cmap[np.linspace(0, 255, self.video.number_of_animals, dtype=int)]
        )
        self.cmap = [QColor(*color) for color in cmap]
        self.cmap_alpha = [QColor(*color, alpha=77) for color in cmap]

        self.id_groups.load_groups(self.video.identities_groups)
        self.video_player.update()

    def click_on_canvas(self, button: int, xdata: float, ydata: float):
        self.selected_fragment = -1
        self.selected_id = -1
        self.selected_blob = None

        for blob in self.blobs.blobs_in_video[self.frame_number]:
            if not blob.bbox_contains_point((xdata, ydata)):
                continue
            for id, centroid in zip(blob.final_identities, blob.final_centroids):
                if id is None:
                    continue
                dist = (centroid[0] - xdata) ** 2 + (centroid[1] - ydata) ** 2
                if dist < SELECT_POINT_DIST:
                    self.selected_id = id
                    self.selected_blob = blob
                    break

        if self.selected_id == -1:
            for blob in self.blobs.blobs_in_video[self.frame_number]:
                if blob.contour_contains_point((xdata, ydata)):
                    self.selected_blob = blob
                    break

            if self.selected_blob is not None:
                if len(
                    self.selected_blob.final_identities
                ) == 1 and self.selected_blob.final_identities[0] not in (0, None):
                    self.selected_id = self.selected_blob.final_identities[0]
                else:
                    self.selected_fragment = (
                        -1
                        if self.selected_blob is None
                        else self.selected_blob.fragment_identifier
                    )
        if self.selected_id not in (None, -1):
            self.id_groups.selected_id(self.selected_id)
        self.frame_number = -1  # this makes info_widget to update
        self.video_player.update()

    def double_click_on_canvas(self, button: int, xdata: float, ydata: float):
        if self.selected_id != -1:
            assert self.selected_blob is not None
            new_id = self.select_id_dialog.exec_with_description(
                "Select the new identity", default=self.selected_id
            )
            if new_id is not None:
                print("change id", self.selected_id, new_id)
                self.selected_blob.update_identity(self.selected_id, new_id)
                self.selected_blob.propagate_identity(self.selected_id, new_id)

        elif self.selected_fragment != -1:
            print("change fragment", self.selected_fragment)

    def update_right_bar(self, blob: Blob | None):
        if self.selected_fragment != -1:
            self.following_label.setText(f"Following fragment {self.selected_fragment}")
        elif self.selected_id != -1:
            self.following_label.setText(f"Following identity {self.selected_id}")
        else:
            self.following_label.setText("")

        self.info_widget.clear()
        if blob is not None:
            self.info_widget.addItems(str(blob).splitlines())

    def paint(self, painter: CustomQPainter, frame_number: int, frame: np.ndarray):

        if self.id_groups.is_active():
            cmap, cmap_alpha = self.id_groups.get_cmaps(self.video.number_of_animals)
        else:
            cmap, cmap_alpha = self.cmap, self.cmap_alpha

        update_info_widget = frame_number != self.frame_number
        self.frame_number = frame_number

        selected_blob = paintBlobs(
            self.view_contours.isChecked(),
            self.view_centroids.isChecked(),
            self.view_bboxes.isChecked(),
            self.view_labels.isChecked(),
            painter,
            self.blobs.blobs_in_video,
            frame_number,
            self.segments,
            cmap,
            cmap_alpha,
            self.selected_fragment,
            self.selected_id,
        )

        if update_info_widget:
            self.update_right_bar(selected_blob)

    def processed_keyPressEvent(self, key: int):
        self.video_player.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.video_player.redirect_keyReleaseEvent(key)
