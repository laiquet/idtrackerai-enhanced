import logging
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import toml
from qtpy.QtCore import Qt, QThread, QTimer, Signal  # type: ignore
from qtpy.QtGui import QAction, QCloseEvent, QColor, QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from idtrackerai import Blob, Fragment, ListOfBlobs, ListOfFragments, Video
from idtrackerai.postprocess import (
    convert_trajectories_file_to_csv_and_json,
    produce_output_dict,
)
from idtrackerai.utils import resolve_path
from idtrackerai_GUI_tools import (
    CanvasMouseEvent,
    CanvasPainter,
    GUIBase,
    LabelRangeSlider,
    QHLine,
    VideoPlayer,
    build_ROI_patches_from_list,
    get_cmap,
)

from .validator_widgets import (
    AdditionalInfo,
    ErrorsExplorer,
    IdGroups,
    IdLabels,
    Interpolator,
    MarkBlobs,
    SetupPoints,
    find_selected_blob,
    paintBlobs,
    paintTrails,
)

SELECT_POINT_DIST = 300


class WarningRedirector(logging.Handler):
    def __init__(self, parent: GUIBase):
        super().__init__()
        self.parent = parent
        self.setLevel(logging.WARNING)

    def emit(self, record: logging.LogRecord):
        QMessageBox.warning(
            self.parent, "Idtracker.ai validator warning", record.getMessage()
        )


class DblClickDialog(QDialog):
    class Answers(Enum):
        Cancel = 0
        ChangeId = 1
        Interpolate = 2
        Reset = 3

    def __init__(self, parent: QWidget, n_animals: int):
        super().__init__(parent)
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(-1)
        self.spinbox.setMaximum(n_animals)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.description = QLabel("0 means null identity")
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
        reset_id_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_BrowserReload), "Reset id"
        )
        self.interp_btn = QPushButton("Interpolate\nhere")
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.interp_btn)
        btn_row.addWidget(reset_id_btn)
        btn_row.addWidget(change_id_btn)
        change_id_btn.setDefault(True)

        cancel_btn.clicked.connect(lambda: self.done(self.Answers.Cancel.value))
        change_id_btn.clicked.connect(lambda: self.done(self.Answers.ChangeId.value))
        reset_id_btn.clicked.connect(lambda: self.done(self.Answers.Reset.value))
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

        return answer, self.spinbox.value(), self.propagate.isChecked()


class LoadSessionObjects(QThread):
    """Independent thread to load lists of Blobs/Fragments
    because they take too long for large sessions."""

    blobs: ListOfBlobs | None = None
    fragments: list[Fragment] | None = None

    def __init__(self, video: Video, parent: QWidget):
        super().__init__(parent)
        self.video = video
        self.parienta = parent

    def run(self):
        for path in (
            self.video.blobs_path_validated,
            self.video.blobs_no_gaps_path,
            self.video.blobs_path,
        ):
            try:
                self.blobs = ListOfBlobs.load(path)
                break
            except FileNotFoundError:
                pass
        else:
            logging.warning(
                "List of blobs not found in %s", self.video.blobs_path.parent
            )
            self.blobs = None

        try:
            self.fragments = ListOfFragments.load(
                self.video.fragments_path, reconnect=False
            ).fragments
            for index, fragment in enumerate(self.fragments):
                if fragment.identifier != index:
                    logging.warning(
                        "Loading an old session, invalid list of fragments format"
                    )
                    raise FileExistsError
        except FileNotFoundError:
            self.fragments = None


class ValidationGUI(GUIBase):
    blobs: ListOfBlobs

    def __init__(self, session_path: Path | None = None):
        super().__init__()

        # TODO logging.getLogger().addHandler(WarningRedirector(self))

        self.setWindowTitle("idtracker.ai | Validation GUI")
        self.documentation_url = (
            "https://idtracker.ai/en/latest/user_guide/validator.html"
        )

        self.video_player = VideoPlayer(self)
        self.video_player.limit_framerate.setChecked(True)
        self.widgets_to_close.append(self.video_player)

        self.video_player.canvas.click_event.connect(self.click_on_canvas)
        self.video_player.canvas.double_click_event.connect(self.double_click_on_canvas)

        def new_changes():
            self.unsaved_changes = True

        self.id_groups = IdGroups(self)
        self.id_groups.needToDraw.connect(self.video_player.update)
        self.id_groups.unsaved_changes.connect(new_changes)

        self.errorsExplorer = ErrorsExplorer()
        self.errorsExplorer.go_to_error.connect(self.go_to_error)

        self.mark_blobs = MarkBlobs(self)
        self.mark_blobs.needToDraw.connect(self.video_player.update)

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

        self.additional_info = AdditionalInfo()

        tabs = QTabWidget()
        tabs.addTab(self.id_groups, "Groups")
        tabs.addTab(self.id_labels, "Labels")
        tabs.addTab(self.setup_points, "Setup Points")
        tabs.addTab(self.mark_blobs, "Mark blobs")
        tabs.setMinimumWidth(250)
        tabs.currentChanged.connect(self.video_player.update)
        right_splitter.addWidget(tabs)
        right_splitter.addWidget(self.additional_info)
        self.interpolator.enabled_changed.connect(
            lambda enabled: tabs.setEnabled(not enabled)
        )

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
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes((1, 10, 1))
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

        self.reset_action = QAction("Reset session...", self)
        self.reset_action.setShortcut("Ctrl+R")
        self.reset_action.setEnabled(False)
        self.reset_action.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
        )
        self.reset_action.triggered.connect(self.reset_session)
        session_menu.addAction(self.reset_action)

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

        find_identity_action = QAction("Find identity", self)
        find_identity_action.setShortcut("Ctrl+F")
        drawing_flags.addSeparator()
        drawing_flags.addAction(find_identity_action)
        find_identity_action.triggered.connect(self.find_identity)

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

        if session_path is not None:
            QTimer.singleShot(0, lambda: self.open_session(session_path))
        self.unsaved_changes = False

    def find_identity(self):
        """Displays a QInputDialog to select an identity to, then, find
        its blob, select it and center the video canvas to its centroid"""
        to_find, success = QInputDialog.getText(
            self,
            "",
            "Identity to find:",
            text=str(self.selected_id) if self.selected_id not in (None, -1) else "",
            flags=Qt.WindowType.SplashScreen,
        )
        to_find = to_find.strip()

        if not success or not to_find:
            return

        if to_find.isdigit():
            identity_to_find = int(to_find)
        else:
            try:
                identity_to_find = self.id_labels.labels.index(to_find)
            except ValueError:
                QMessageBox.warning(
                    self, "Find error", f'Identity not recognized: "{to_find}"'
                )
                return

        for blob in self.blobs.blobs_in_video[self.current_frame_number]:
            for identity, centroid in blob.final_ids_and_centroids:
                if identity == identity_to_find:
                    self.selected_blob = blob
                    self.selected_id = identity
                    self.selection_last_location = centroid
                    self.current_frame_number = -1  # this makes info_widget to update
                    self.video_player.center_canvas_at(
                        *centroid, 50 * self.median_speed
                    )
                    return

        QMessageBox.warning(
            self, "Find error", f"Identity {identity_to_find} not found in this frame"
        )

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.id_groups.uncheck_edit_buttons()
            self.setup_points.add.setChecked(False)

    def go_to_error(
        self, kind: str, start: int, length: int, where: np.ndarray, identity: int
    ):
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
            where = where[0]
        else:
            # Set the zoom to view ~50 time steps in the current canvas width
            self.video_player.center_canvas_at(
                *where, zoom_scale=50 * self.median_speed
            )
        self.selection_last_location = None if np.isnan(where).any() else tuple(where)

        self.selected_id = identity
        if kind in ("Jump", "Miss id"):
            self.interpolator.set_interpolation_params(identity, start, start + length)
        else:
            self.interpolator.setActivated(False)
        self.video_player.setCurrentFrame(
            start - 1 if start > 0 and kind in ("Jump", "Miss id") else start, True
        )

    def reset_session(self):
        start, finish = self.reset_session_dialog.exec()
        if start is not None:
            self.blobs.reset_user_generated_corrections(start, finish)
            self.errorsExplorer.non_accepted_jumps[start:finish] = True
            self.update_trajectories_range(start, finish)

    def save_session(self):
        self.video.identities_labels = self.id_labels.get_labels()[1:]
        self.video.identities_groups = self.id_groups.get_groups()
        self.video.setup_points = self.setup_points.get_points()
        self.video.save()
        self.blobs.save(self.video.blobs_path_validated)

        progress = QProgressDialog(
            "Computing trajectories",
            "Abort",
            0,
            self.video.number_of_frames + 1,
            self,
            Qt.WindowType.SplashScreen,
        )
        progress.setMinimumDuration(1500)
        progress.setModal(True)

        self.save_thread = SaveTrajectoriesThread(
            self.blobs.blobs_in_video, self.video, self.fragments
        )
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
            video = self.video
        except FileNotFoundError as err:
            QMessageBox.warning(self, "Loading session error", str(err))
            return

        if hasattr(video, "general_timer") and not video.general_timer.finished:
            answer = QMessageBox.question(
                self,
                "Loading session warning",
                (
                    "The session you are trying to load has not finished, unexpected"
                    " behavior can happen. Do you want to continue?"
                ),
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            )
            if answer != QMessageBox.StandardButton.Ok:
                return

        loading_thread = LoadSessionObjects(video, self)
        progress_bar = QProgressDialog(
            "Loading session, please wait...",
            "Close app",
            0,
            0,
            self,
            Qt.WindowType.SplashScreen,
        )
        progress_bar.setMinimumDuration(100)
        progress_bar.canceled.connect(loading_thread.terminate)
        progress_bar.canceled.connect(sys.exit)
        progress_bar.setModal(True)

        loading_thread.start()
        while loading_thread.isRunning():
            QApplication.processEvents()
        progress_bar.cancel()

        if loading_thread.blobs is None:
            QMessageBox.warning(
                self, "Loading session error", "List of blobs not found"
            )
            return
        self.blobs = loading_thread.blobs
        self.fragments = loading_thread.fragments

        self.additional_info.fragments = self.fragments

        self.video_player.update_video_paths(
            video.video_paths,
            video.number_of_frames,
            (video.original_width, video.original_height),
            video.frames_per_second,
            res_reduct=video.resolution_reduction,
        )
        self.n_animals = video.number_of_animals
        self.n_frames = video.number_of_frames
        self.generate_trajectories(self.blobs.blobs_in_video)
        self.median_speed = np.nanmedian(
            np.sqrt(np.sum(np.diff(self.trajectories, axis=0) ** 2, axis=-1))
        )
        self.centralWidget().setEnabled(True)
        self.dbl_click_dialog = DblClickDialog(self, video.number_of_animals)

        cmap = [(255, 255, 255)] + (
            get_cmap()[np.linspace(0, 255, video.number_of_animals, dtype=int)].tolist()
        )
        self.cmap = tuple(QColor(*color) for color in cmap)
        self.cmap_alpha = tuple(QColor(*color, alpha=77) for color in cmap)

        self.id_groups.load_groups(video.identities_groups)
        self.id_labels.load_labels(
            video.identities_labels
            or [str(i + 1) for i in range(video.number_of_animals)]
        )

        self.setup_points.load_points(video.setup_points)
        self.errorsExplorer.set_references(
            self.trajectories,
            self.unidentified,
            self.duplicated,
            self.blobs,
            video.tracking_intervals,
        )
        self.interpolator.set_references(
            self.trajectories, self.unidentified, self.duplicated, self.blobs
        )
        self.video_player.update()
        self.unsaved_changes = False

        if hasattr(video, "ROI_list") and video.ROI_list:
            self.view_ROIs.setEnabled(True)
            self.view_ROIs.setChecked(True)
            self.ROI_pathces = build_ROI_patches_from_list(
                video.ROI_list,
                video.resolution_reduction,
                video.original_width,
                video.original_height,
            )
        else:
            self.view_ROIs.setChecked(False)
            self.view_ROIs.setEnabled(False)

        self.setWindowTitle("Validator | " + video.session_folder.name)

        self.save_action.setEnabled(True)
        self.reset_action.setEnabled(True)

        self.reset_session_dialog = ResetSessionDialog(self, video.number_of_frames)

    def click_on_canvas(self, event: CanvasMouseEvent):
        self.selected_blob, self.selected_id, self.selection_last_location = clicked_id(
            self.blobs.blobs_in_video[self.current_frame_number], event
        )
        self.id_groups.selected_id(self.selected_id)
        self.current_frame_number = -1  # this makes info_widget to update
        self.video_player.update()

    def double_click_on_canvas(self, event: CanvasMouseEvent):
        if (
            self.selected_blob is None
            or self.id_groups.is_active()
            or event.button != Qt.MouseButton.LeftButton
        ):
            return

        if self.selection_last_location is None:
            # clicked on a blob without centroids
            answer, new_id, propagate = self.dbl_click_dialog.exec_with_description(0)
            if answer == DblClickDialog.Answers.ChangeId:
                self.selected_blob.add_centroid(event.xy_data, new_id)
                self.update_trajectories_range(self.current_frame_number)
            return

        # clicked on a blob with centroid
        answer, new_id, propagate = self.dbl_click_dialog.exec_with_description(
            self.interpolator.animal_id + 1
            if self.interpolator.isEnabled()
            else self.selected_id
        )

        if answer == DblClickDialog.Answers.Reset:
            new_id = None
            answer = DblClickDialog.Answers.ChangeId

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

    def paint(self, painter: CanvasPainter, frame_number: int):
        blobs_in_frame = self.blobs.blobs_in_video[frame_number]
        if self.id_groups.is_active():
            cmap, cmap_alpha = self.id_groups.get_cmaps(self.video.number_of_animals)
        else:
            cmap, cmap_alpha = self.cmap, self.cmap_alpha

        update_info_widget = frame_number != self.current_frame_number
        self.current_frame_number = frame_number

        if self.selected_blob is not None and self.selected_blob not in blobs_in_frame:
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
            self.mark_blobs(blobs_in_frame, self.fragments),
        )

        if self.setup_points.isVisible():
            self.setup_points.paint_on_canvas(painter)

        if self.interpolator.isEnabled():
            self.interpolator.paint_on_canvas(painter, frame_number)

        if update_info_widget:
            self.additional_info.set_data(self.selected_blob, len(blobs_in_frame))

    def closeEvent(self, event: QCloseEvent):
        if not self.unsaved_changes:
            return super().closeEvent(event)

        answer = QMessageBox.question(
            self,
            "Save changes?",
            (
                "There are unsaved changes. Changes which are not saved will be"
                " permanently lost."
            ),
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

    def update_trajectories_range(self, start: int, finish: int | None = None):
        finish = start + 1 if finish is None else finish
        ids_in_frame = set()
        self.trajectories[start:finish] = np.nan
        self.duplicated[start:finish] = False
        self.unidentified[start:finish] = False
        for blobs_in_frame in self.blobs.blobs_in_video[start:finish]:
            ids_in_frame.clear()
            for blob in blobs_in_frame:
                for identity, centroid in blob.final_ids_and_centroids:
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

        progress_bar = QProgressDialog(
            "Analyzing trajectories",
            "Close app",
            0,
            number_of_frames - 1,
            self,
            Qt.WindowType.SplashScreen,
        )
        progress_bar.setMinimumDuration(1000)
        progress_bar.canceled.connect(sys.exit)
        progress_bar.setModal(True)

        for index, blobs_in_frame in enumerate(blobs_in_video):
            progress_bar.setValue(index)
            ids_in_frame.clear()
            for blob in blobs_in_frame:
                for identity, centroid in blob.final_ids_and_centroids:
                    if identity not in (None, 0):
                        self.trajectories[blob.frame_number, identity - 1] = centroid
                        if identity in ids_in_frame:
                            self.duplicated[blob.frame_number, identity - 1] = True
                        ids_in_frame.add(identity)
                    else:
                        self.unidentified[blob.frame_number] = True


def clicked_id(
    blobs: list[Blob], click: CanvasMouseEvent
) -> tuple[Blob | None, int | None, tuple[float, float] | None]:
    distances_to_centroids: list[
        tuple[Blob, int | None, tuple[float, float], float]
    ] = []

    for blob in blobs:
        if blob.contains_point(click.xy_data):
            for identity, centroid in blob.final_ids_and_centroids:
                dist = click.sq_distance_to(centroid)
                distances_to_centroids.append((blob, identity, centroid, dist))
            if not distances_to_centroids:  # blob with no centroids
                return blob, -1, None
            break

    if distances_to_centroids:
        return min(distances_to_centroids, key=lambda x: x[-1])[:-1]

    for blob in blobs:
        for identity, centroid in blob.final_ids_and_centroids:
            dist = click.sq_distance_to(centroid)
            if dist < (SELECT_POINT_DIST * click.zoom):
                distances_to_centroids.append((blob, identity, centroid, dist))

    if distances_to_centroids:
        return min(distances_to_centroids, key=lambda x: x[-1])[:-1]

    return None, -1, None


class SaveTrajectoriesThread(QThread):
    progress_changed = Signal(int)

    def __init__(
        self,
        blobs_in_video: list[list[Blob]],
        video: Video,
        list_of_fragments: list[Fragment] | None,
    ):
        super().__init__()
        self.blobs_in_video = blobs_in_video
        self.fragments = list_of_fragments
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
            self.fragments,
            progress_bar=self.progress_changed,
            abort=lambda: self.abort,
        )
        if self.abort:
            return
        trajectories_file = (
            self.video.trajectories_folder / "trajectories_validated.npy"
        )
        logging.info("Saving trajectories at %s", trajectories_file)
        np.save(trajectories_file, trajectories)  # type: ignore

        if (self.video.trajectories_folder / "trajectories").is_dir():
            convert_trajectories_file_to_csv_and_json(trajectories_file)

        self.progress_changed.emit(self.video.number_of_frames)
        self.success = True

    def quit(self):
        self.abort = True


class ResetSessionDialog(QDialog):
    """Pop up to select the range of the user corrections reset.
    Reset is activated in the menu bar / Session / Reset session"""

    class Answers(Enum):
        Cancel = 0
        RangeReset = 1
        AllReset = 2

    def __init__(self, parent, n_frames: int):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(
            QLabel(
                '<span style="font-weight:600">Resetting removes all user corrections'
            ),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(
            QLabel("Range of frames to reset:"), alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.double_slider = LabelRangeSlider(0, n_frames, parent)
        layout.addWidget(self.double_slider)
        btn_layout = QHBoxLayout()
        style = self.style()
        cancel_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogCancelButton), "Cancel"
        )
        range_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_BrowserReload), "Reset range"
        )
        all_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_BrowserReload), "Reset all"
        )
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(range_btn)
        btn_layout.addWidget(all_btn)
        layout.addLayout(btn_layout)

        cancel_btn.clicked.connect(lambda: self.done(self.Answers.Cancel.value))
        range_btn.clicked.connect(lambda: self.done(self.Answers.RangeReset.value))
        all_btn.clicked.connect(lambda: self.done(self.Answers.AllReset.value))

        cancel_btn.setAutoDefault(False)
        range_btn.setAutoDefault(False)
        all_btn.setAutoDefault(False)

    def exec(self) -> tuple[int | None, int | None]:
        match self.Answers(super().exec()):
            case self.Answers.Cancel:
                return None, None
            case self.Answers.RangeReset:
                return self.double_slider.value()
            case self.Answers.AllReset:
                return self.double_slider.minimum(), self.double_slider.maximum()
