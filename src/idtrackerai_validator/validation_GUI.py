import logging
from enum import Enum
from pathlib import Path

import numpy as np
import toml
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent, QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from rich.progress import track

from idtrackerai import Blob, ListOfBlobs, Video
from idtrackerai.postprocess import (
    convert_trajectories_file_to_csv_and_json,
    produce_output_dict,
)
from idtrackerai.utils import resolve_path
from idtrackerai_GUI_tools import CustomPainter, GUIBase, QHLine, VideoPlayer
from idtrackerai_GUI_tools import __file__ as idtrackerai_GUI_tools_file
from idtrackerai_GUI_tools import build_ROI_patches_from_list, key_event_modifier

from .validator_widgets import (
    ErrorsExplorer,
    IdGroups,
    IdLabels,
    Interpolator,
    SetupPoints,
    find_selected_blob,
    paintBlobs,
    paintTrails,
)

parent_dir = Path(idtrackerai_GUI_tools_file).parent
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
SELECT_POINT_DIST = 300


class DblClickDialog(QDialog):
    class Answers(Enum):
        Cancel = 0
        ChangeId = 1
        Interpolate = 2

    # TODO pop up dialog when closing without saving
    def __init__(self, parent: QWidget, n_animals: int):
        super().__init__(parent)
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(-1)
        self.spinbox.setMaximum(n_animals)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.description = QLabel(
            "0 means null identity and -1 means\nto return to assigned identity"
        )
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setWordWrap(True)

        self.propagate = QCheckBox("Propagate identity")
        self.propagate.setChecked(True)
        spin_row = QHBoxLayout()
        spin_row.addWidget(QLabel("New identity:"))
        spin_row.addWidget(self.spinbox)
        spin_row.addWidget(self.propagate)
        btn_row = QHBoxLayout()

        style = self.style()
        cancel_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogCancelButton), "Cancel"
        )
        change_id_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogOkButton), "Change id"
        )
        self.interp_btn = QPushButton("Interpolate\nhere")
        btn_row.addWidget(self.interp_btn)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(change_id_btn)
        change_id_btn.setDefault(True)

        cancel_btn.clicked.connect(lambda: self.done(self.Answers.Cancel.value))
        change_id_btn.clicked.connect(lambda: self.done(self.Answers.ChangeId.value))
        self.interp_btn.clicked.connect(
            lambda: self.done(self.Answers.Interpolate.value)
        )

        main_layout.addLayout(spin_row)
        main_layout.addWidget(self.description)
        main_layout.addLayout(btn_row)

    def exec_with_description(
        self, default: int | None
    ) -> tuple[Answers, int | None, bool]:
        if default is not None:
            self.spinbox.setValue(default)
        self.interp_btn.setEnabled(default is not None and default > 0)

        self.spinbox.setFocus()
        answer = self.Answers(super().exec())

        new_id = self.spinbox.value()
        if new_id == -1:
            new_id = None
        return answer, new_id, self.propagate.isChecked()


class CustomListWidget(QListWidget):
    def keyPressEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyReleaseEvent(event)


class ValidationGUI(GUIBase):
    def __init__(self, session_path: Path | None = None):
        super().__init__()

        self.setWindowTitle("idTracker.ai | Validation GUI")
        self.documentation_url = "https://idtrackerai.readthedocs.io/en/latest/"

        self.video_player = VideoPlayer(self)
        self.video_player.limit_framerate.setChecked(True)
        self.widgets_to_close.append(self.video_player)

        self.video_player.canvas.click_event.connect(self.click_on_canvas)
        self.video_player.canvas.double_click_event.connect(self.double_click_on_canvas)

        def new_changes():
            self.unsaved_changes = True

        self.info_widget = CustomListWidget()
        self.info_widget.setAlternatingRowColors(True)
        self.following_label = QLabel()
        self.following_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.id_groups = IdGroups(self)
        self.id_groups.needToDraw.connect(self.video_player.update)
        self.id_groups.unsaved_changes.connect(new_changes)

        self.errorsExplorer = ErrorsExplorer()
        self.errorsExplorer.go_to_error.connect(self.go_to_error)

        self.interpolator = Interpolator()
        self.interpolator.neew_to_draw.connect(self.video_player.update)
        self.interpolator.update_trajectories.connect(self.update_trajectories_range)
        self.interpolator.go_to_frame.connect(self.video_player.setCurrentFrame)
        self.interpolator.preload_frames.connect(self.video_player.preload_frames)
        self.interpolator.interpolation_accepted.connect(
            self.errorsExplorer.accepted_interpolation
        )

        self.id_labels = IdLabels()
        self.id_labels.needToDraw.connect(self.video_player.update)
        self.id_labels.needToDraw.connect(new_changes)

        self.setup_points = SetupPoints()
        self.setup_points.needToDraw.connect(self.video_player.update)
        self.setup_points.needToDraw.connect(new_changes)

        self.video_player.canvas.click_event.connect(self.setup_points.click_event)
        self.video_player.canvas.click_event.connect(self.interpolator.click_event)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setContentsMargins(8, 0, 0, 0)
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 8)
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        info_layout.addWidget(self.following_label)
        info_layout.addWidget(self.info_widget)

        tabs = QTabWidget()
        tabs.addTab(self.id_groups, "Groups")
        tabs.addTab(self.id_labels, "Labels")
        tabs.addTab(self.setup_points, "Setup Points")
        tabs.currentChanged.connect(self.video_player.update)
        right_splitter.addWidget(tabs)
        right_splitter.addWidget(info_widget)

        left_splitter = QVBoxLayout()
        left_splitter.addWidget(self.errorsExplorer)
        left_splitter.addWidget(QHLine())
        left_splitter.addWidget(self.interpolator)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.video_player.layout().setContentsMargins(8, 0, 8, 0)
        left_widget = QWidget()
        left_widget.setLayout(left_splitter)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.video_player)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        self.centralWidget().layout().addWidget(splitter)
        self.centralWidget().setEnabled(False)
        self.centralWidget().layout().setContentsMargins(8, 0, 8, 8)

        self.selected_id: int | None = None
        self.selected_blob: Blob | None = None
        self.selection_last_location: tuple[float, float] | None = None

        self.video_player.painting_time.connect(self.paint)
        self.current_frame_number = -1
        self.trajectories: np.ndarray
        """Float, positions of animals"""
        self.unidentified: np.ndarray
        """Bool, there is some identity without centroid"""
        self.duplicated: np.ndarray
        """Bool, some centroid have the same identity"""

        session_menu = self.menuBar().addMenu("Session")

        open_action = QAction("Open session", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton)
        )
        open_action.triggered.connect(
            lambda: self.open_session(
                QFileDialog.getExistingDirectory(
                    self, "Open session directory", ".", QFileDialog.Option.ShowDirsOnly
                )
            )
        )
        session_menu.addAction(open_action)

        self.save_action = QAction("Save session", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        self.save_action.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton)
        )
        self.save_action.triggered.connect(self.save_session)
        session_menu.addAction(self.save_action)

        drawing_flags = self.menuBar().addMenu("Draw")

        self.view_labels = QAction("Labels", self)
        self.view_labels.setShortcut("Alt+L")
        self.view_contours = QAction("Contours", self)
        self.view_contours.setShortcut("Alt+C")
        self.view_centroids = QAction("Centroids", self)
        self.view_centroids.setShortcut("Alt+P")
        self.view_bboxes = QAction("Bounding boxes", self)
        self.view_bboxes.setShortcut("Alt+B")
        self.view_trails = QAction("Trails", self)
        self.view_trails.setShortcut("Alt+T")
        self.view_ROIs = QAction("Regions of interest", self)
        self.view_ROIs.setShortcut("Alt+R")

        drawing_flags.addActions(
            (
                self.view_labels,
                self.view_contours,
                self.view_centroids,
                self.view_bboxes,
                self.view_trails,
                self.view_ROIs,
            )
        )

        for action in drawing_flags.actions():
            action.setCheckable(True)
            action.toggled.connect(self.video_player.update)

        # Defaults
        self.view_labels.setChecked(True)
        self.view_contours.setChecked(True)
        self.view_centroids.setChecked(True)
        self.view_bboxes.setChecked(False)
        self.view_trails.setChecked(True)
        self.view_ROIs.setChecked(False)

        tooltips = toml.load(Path(__file__).parent / "tooltips.toml")

        self.interpolator.apply_btn.setToolTip(tooltips["apply_interpolation"])
        self.interpolator.abort_btn.setToolTip(tooltips["abort_interpolation"])
        self.errorsExplorer.jumps_th.setToolTip(tooltips["jumps_th"])
        self.errorsExplorer.reset_jumps.setToolTip(tooltips["reset_jumps"])
        self.errorsExplorer.jumps_th_label.setToolTip(tooltips["jumps_th"])
        self.errorsExplorer.jumps_th.setToolTip(tooltips["jumps_th"])
        self.errorsExplorer.update_btn.setToolTip(tooltips["update_errors"])
        self.interpolator.interpolation_order_box.setToolTip(
            tooltips["interpolation_order"]
        )
        self.interpolator.interpolation_order_label.setToolTip(
            tooltips["interpolation_order"]
        )
        for index in range(self.interpolator.input_size_row.count()):
            self.interpolator.input_size_row.itemAt(index).widget().setToolTip(
                tooltips["input_size"]
            )

        self.center_window()
        if session_path is not None:
            QTimer.singleShot(0, lambda: self.open_session(session_path))
        self.unsaved_changes = False

    def go_to_error(
        self,
        kind: str,
        start: int,
        length: int,
        where: np.ndarray | None,
        identity: int,
    ):
        if where is None:
            where = self.trajectories[start]
            assert where is not None

        if where.ndim == 2:
            # Set the zoom to capture all positions of 'where'
            xmax, ymax = np.nanmax(where, axis=0)
            xmin, ymin = np.nanmin(where, axis=0)
            zoom_scale = max(
                30 * self.median_speed, 1.8 * (xmax - xmin), 1.8 * (ymax - ymin)
            )
            self.video_player.center_canvas_at(
                0.5 * (xmax + xmin), 0.5 * (ymin + ymax), zoom_scale=zoom_scale
            )
            self.selection_last_location = (
                None if np.any(np.isnan(where[0])) else tuple(where[0])
            )
        else:
            # Set the zoom to view ~50 time steps in the current canvas width
            self.video_player.center_canvas_at(
                *where, zoom_scale=50 * self.median_speed
            )
            self.selection_last_location = (
                None if np.any(np.isnan(where)) else tuple(where)
            )

        self.selected_id = identity
        if kind in ("Jump", "Miss id"):
            # TODO start preloading frames somehow
            self.interpolator.set_interpolation_params(identity, start, start + length)
        else:
            self.interpolator.setActivated(False)
        self.video_player.setCurrentFrame(
            start - 1 if start > 0 and kind in ("Jump", "Miss id") else start, True
        )

    def save_session(self):
        self.video.identities_labels = self.id_labels.get_labels()[1:]
        self.video.identities_groups = self.id_groups.get_groups()
        self.video.setup_points = self.setup_points.get_points()
        self.video.save()
        self.blobs.save(self.video.blobs_path_validated)

        progress = QProgressDialog(
            "Computing trajectories", "Abort", 0, self.video.number_of_frames + 1, self
        )
        progress.setMinimumDuration(1500)
        progress.setModal(True)

        self.save_thread = SaveTrajectoriesThread(self.blobs.blobs_in_video, self.video)
        progress.canceled.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.finish_saving)
        self.save_thread.progress_changed.connect(progress.setValue)
        self.save_thread.start()

    def finish_saving(self):
        if self.save_thread.success:
            self.unsaved_changes = False

    def open_session(self, session_path: Path | str):
        if not session_path:
            return
        session_path = resolve_path(session_path)
        try:
            self.video = Video.load(session_path)
        except FileNotFoundError as err:
            QMessageBox.warning(self, "Loading session error", str(err))
            return

        if (
            hasattr(self.video, "general_timer")
            and not self.video.general_timer.finished
        ):
            answer = QMessageBox.warning(
                self,
                "Loading session warning",
                "The session you are trying to load has not finished, unexpected behavior can happen.",
                QMessageBox.StandardButton.Abort | QMessageBox.StandardButton.Ignore,
            )
            if answer == QMessageBox.StandardButton.Abort:
                return

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
        self.dbl_click_dialog = DblClickDialog(self, self.video.number_of_animals)

        cmap = [(255, 255, 255)] + list(
            general_cmap[np.linspace(0, 255, self.video.number_of_animals, dtype=int)]
        )
        self.cmap = [QColor(*color) for color in cmap]
        self.cmap_alpha = [QColor(*color, alpha=77) for color in cmap]

        self.id_groups.load_groups(self.video.identities_groups)
        self.id_labels.load_labels(self.video.identities_labels)
        self.setup_points.load_points(self.video.setup_points)
        self.errorsExplorer.set_references(
            self.trajectories,
            self.unidentified,
            self.duplicated,
            self.video.tracking_intervals,
        )
        self.interpolator.set_references(
            self.trajectories, self.unidentified, self.duplicated, self.blobs
        )
        self.video_player.update()
        self.unsaved_changes = False

        if hasattr(self.video, "ROI_list") and self.video.ROI_list:
            self.view_ROIs.setEnabled(True)
            self.ROI_pathces = build_ROI_patches_from_list(
                self.video.width, self.video.height, self.video.ROI_list
            )
        else:
            self.view_ROIs.setChecked(False)
            self.view_ROIs.setEnabled(False)

        self.setWindowTitle(
            "idtracker.ai validator | " + self.video.session_folder.name
        )

        self.save_action.setEnabled(True)

    def click_on_canvas(self, button: int, zoom: float, x: float, y: float):
        self.selected_blob, self.selected_id, self.selection_last_location = clicked_id(
            self.blobs.blobs_in_video[self.current_frame_number], x, y, zoom
        )
        if self.selected_id not in (-1, None):
            self.id_groups.selected_id(self.selected_id)
        self.current_frame_number = -1  # this makes info_widget to update
        self.video_player.update()

    def double_click_on_canvas(self, button: int, zoom: float, x: float, y: float):
        if (
            self.selected_blob is None
            or self.id_groups.is_active()
            or button != Qt.MouseButton.LeftButton
        ):
            return

        if self.selection_last_location is not None:
            # clicked on a blob with centroid
            assert self.selection_last_location is not None
            answer, new_id, propagate = self.dbl_click_dialog.exec_with_description(
                self.selected_id
            )
            if answer == DblClickDialog.Answers.ChangeId:
                self.selected_blob.update_identity(
                    self.selected_id, new_id, self.selection_last_location
                )
                if propagate:
                    lower, upper = self.selected_blob.propagate_identity(
                        self.selected_id, new_id, self.selection_last_location
                    )
                    QMessageBox.information(
                        self,
                        "Identification change",
                        f"Identification propagated from frame {lower} to frame {upper}",
                    )
                    self.update_trajectories_range(lower, upper + 1)
                else:
                    self.update_trajectories_range(self.current_frame_number)
                return
            if answer == DblClickDialog.Answers.Interpolate:
                assert self.selected_id is not None and self.selected_id > 0
                self.interpolator.set_interpolation_params(
                    self.selected_id,
                    self.current_frame_number,
                    self.current_frame_number + 1,
                )

        else:
            # clicked on a blob without centroids
            answer, new_id, propagate = self.dbl_click_dialog.exec_with_description(0)
            if answer == DblClickDialog.Answers.ChangeId:
                self.selected_blob.add_centroid((x, y), new_id)

    def update_right_bar(self, blob: Blob | None):
        self.info_widget.clear()
        if blob is not None:
            self.info_widget.addItems(str(blob).splitlines())
        self.following_label.setText(
            ""
            if self.selected_id in (-1, None)
            else f"Identity {self.selected_id}: extra info"
        )

    def paint(self, painter: CustomPainter, frame_number: int):
        blobs_in_frame = self.blobs.blobs_in_video[frame_number]
        if self.id_groups.is_active():
            cmap, cmap_alpha = self.id_groups.get_cmaps(self.video.number_of_animals)
        else:
            cmap, cmap_alpha = self.cmap, self.cmap_alpha

        update_info_widget = frame_number != self.current_frame_number
        self.current_frame_number = frame_number

        if self.selected_blob not in blobs_in_frame:
            self.selected_blob, self.selection_last_location = find_selected_blob(
                blobs_in_frame, self.selected_id, self.selection_last_location
            )

        if self.view_trails.isChecked():
            paintTrails(self.current_frame_number, painter, self.trajectories, cmap)

        if self.view_ROIs.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 0, 0, 50))
            painter.drawPath(self.ROI_pathces)

        paintBlobs(
            self.view_contours.isChecked(),
            self.view_centroids.isChecked(),
            self.view_bboxes.isChecked(),
            self.view_labels.isChecked(),
            painter,
            blobs_in_frame,
            cmap,
            cmap_alpha,
            self.selected_blob,
            self.selection_last_location,
            self.id_labels.get_labels(),
        )

        if self.setup_points.isVisible():
            self.setup_points.paint_on_canvas(painter)

        if self.interpolator.isEnabled():
            self.interpolator.paint_on_canvas(painter, frame_number)

        if update_info_widget:
            self.update_right_bar(self.selected_blob)

    def closeEvent(self, event: QCloseEvent):
        if not self.unsaved_changes:
            return super().closeEvent(event)

        answer = QMessageBox.question(
            self,
            "Save changes?",
            "There are unsaved changes. Changes which are not saved will be permanently lost.",
            QMessageBox.StandardButton.Cancel
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Discard:
            return super().closeEvent(event)
        if answer == QMessageBox.StandardButton.Save:
            self.save_session()
            return super().closeEvent(event)
        return event.ignore()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.id_groups.enter_pressed()
            self.setup_points.enter_pressed()
        self.video_player.redirect_keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        self.video_player.redirect_keyReleaseEvent(event)

    def update_trajectories_range(self, start: int, finish: int | None = None):
        finish = start + 1 if finish is None else finish
        ids_in_frame = set()
        self.trajectories[start:finish] = np.nan
        self.duplicated[start:finish] = False
        self.unidentified[start:finish] = False
        for blobs_in_frame in self.blobs.blobs_in_video[start:finish]:
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
                        self.unidentified[blob.frame_number] = True
        self.interpolator.trajectories_have_been_updated()
        self.errorsExplorer.update_list_of_errors()
        self.video_player.update()
        self.unsaved_changes = True

    def generate_trajectories(self, blobs_in_video: list[list[Blob]]):
        number_of_frames = len(blobs_in_video)
        self.trajectories = np.full((number_of_frames, self.n_animals, 2), np.NaN)
        self.unidentified = np.zeros((number_of_frames), bool)
        self.duplicated = np.zeros((number_of_frames, self.n_animals), bool)
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
                        self.unidentified[blob.frame_number] = True


def clicked_id(
    blobs: list[Blob], x, y, zoom: float
) -> tuple[Blob | None, int | None, tuple[float, float] | None]:
    distances_to_centroids: list[
        tuple[Blob, int | None, tuple[float, float], float]
    ] = []

    for blob in blobs:
        if blob.contains_point((x, y)):
            for identity, centroid in zip(blob.final_identities, blob.final_centroids):
                dist = (centroid[0] - x) ** 2 + (centroid[1] - y) ** 2
                distances_to_centroids.append((blob, identity, centroid, dist))
            if not distances_to_centroids:  # blob with no centroids
                return blob, -1, None
            break

    if distances_to_centroids:
        return min(distances_to_centroids, key=lambda x: x[-1])[:-1]

    for blob in blobs:
        for identity, centroid in zip(blob.final_identities, blob.final_centroids):
            dist = (centroid[0] - x) ** 2 + (centroid[1] - y) ** 2
            if dist < (SELECT_POINT_DIST * zoom):
                distances_to_centroids.append((blob, identity, centroid, dist))

    if distances_to_centroids:
        return min(distances_to_centroids, key=lambda x: x[-1])[:-1]

    return None, -1, None


class SaveTrajectoriesThread(QThread):
    progress_changed = pyqtSignal(int)

    def __init__(self, blobs_in_video: list[list[Blob]], video: Video):
        super().__init__()
        self.blobs_in_video = blobs_in_video
        self.video = video
        self.success = False
        self.finished.connect(
            lambda: self.progress_changed.emit(len(self.blobs_in_video) + 1)
        )

    def run(self):
        self.abort = False

        trajectories = produce_output_dict(
            self.blobs_in_video,
            self.video,
            progress_bar=self.progress_changed,
            abort=lambda: self.abort,
        )
        if self.abort:
            return
        trajectories_file = self.video.trajectories_folder / (
            "trajectories_validated.npy"
        )
        logging.info("Saving trajectories at %s", trajectories_file)
        np.save(trajectories_file, trajectories)  # type: ignore

        if (self.video.trajectories_folder / "trajectories").is_dir():
            convert_trajectories_file_to_csv_and_json(trajectories_file)

        self.progress_changed.emit(self.video.number_of_frames)
        self.success = True

    def quit(self):
        self.abort = True
