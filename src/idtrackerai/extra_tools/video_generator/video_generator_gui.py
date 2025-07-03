from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from qtpy.QtCore import QRect, Qt, QUrl
from qtpy.QtGui import QDesktopServices, QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import QToggleSwitch

from idtrackerai.GUI_tools import (
    CanvasPainter,
    GUIBase,
    IdLabels,
    LabelRangeSlider,
    QHLine,
    VideoPlayer,
    get_cmap,
    open_session,
)
from idtrackerai.utils import load_trajectories

from .general_video import draw_general_frame, generate_general_video
from .individual_videos import (
    draw_collage_frame,
    generate_individual_video,
    get_geometries,
    get_individual_miniframes,
)


class VideoGeneratorGUI(GUIBase):

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Video Generator")
        self.documentation_url = (
            "https://idtracker.ai/latest/user_guide/video_generators.html"
        )

        self.video_player = VideoPlayer(self)
        self.video_player.is_muted = True
        self.video_player.draw_in_color.setEnabled(False)
        self.video_player.draw_in_color.setChecked(True)
        self.widgets_to_close.append(self.video_player)
        self.video_player.painting_time.connect(self.preview_frame)

        self.open_button = QPushButton()
        self.open_button.setIcon(QIcon.fromTheme("document-open"))
        self.open_button.setText("Open Session")
        self.open_button.setShortcut("Ctrl+O")
        self.open_button.setToolTip("Open a session to generate videos")
        self.open_button.clicked.connect(self.open)

        video_kind_row = QWidget()
        video_kind_layout = QHBoxLayout()
        video_kind_row.setLayout(video_kind_layout)
        video_kind_layout.setContentsMargins(0, 0, 0, 0)
        video_kind_layout.addWidget(QLabel("General Video"))
        video_kind_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.individual_kind_switch = QToggleSwitch(self)
        self.individual_kind_switch.setChecked(False)
        self.individual_kind_switch.setToolTip(
            "Switch between General Video and Individual Videos"
        )
        self.individual_kind_switch.toggled.connect(self.toggle_video_kind)
        video_kind_layout.addWidget(self.individual_kind_switch)
        video_kind_layout.addWidget(QLabel("Individual Video"))

        self.id_labels = IdLabels()
        self.id_labels.needToDraw.connect(self.video_player.update)

        self.individual_size_row = QWidget(self)
        individual_size_layout = QHBoxLayout()
        self.individual_size_row.setLayout(individual_size_layout)
        individual_size_layout.setContentsMargins(0, 0, 0, 0)
        self.individual_size = QSpinBox(self)
        self.individual_size.setSuffix(" pixels")
        self.individual_size.setRange(2, 1000)
        self.individual_size.setSingleStep(2)
        self.individual_size.valueChanged.connect(self.individual_size_changed)
        individual_size_layout.addWidget(QLabel("Individual Size:"))
        individual_size_layout.addWidget(self.individual_size)
        self.individual_size_row.setVisible(False)

        render_in_gray_row = QWidget(self)
        render_in_gray_layout = QHBoxLayout()
        render_in_gray_row.setLayout(render_in_gray_layout)
        render_in_gray_layout.setContentsMargins(0, 0, 0, 0)
        self.render_in_color = QToggleSwitch()
        self.render_in_color.setChecked(True)
        self.render_in_color.setToolTip(
            "Render the video in color (checked) or grayscale (unchecked)"
        )
        self.render_in_color.toggled.connect(self.video_player.update)
        render_in_gray_layout.addWidget(QLabel("Render in Color:"))
        render_in_gray_layout.addWidget(self.render_in_color)

        self.general_trace_length_row = QWidget(self)
        general_trace_length_layout = QHBoxLayout()
        self.general_trace_length_row.setLayout(general_trace_length_layout)
        general_trace_length_layout.setContentsMargins(0, 0, 0, 0)
        self.general_trace_length = QSpinBox(self)
        self.general_trace_length.setSuffix(" frames")
        self.general_trace_length.setRange(0, 1000)
        self.general_trace_length.setValue(20)
        self.general_trace_length.valueChanged.connect(self.video_player.update)
        general_trace_length_layout.addWidget(QLabel("General Trace Length:"))
        general_trace_length_layout.addWidget(self.general_trace_length)

        self.resize_factor_row = QWidget(self)
        resize_factor_layout = QHBoxLayout()
        self.resize_factor_row.setLayout(resize_factor_layout)
        resize_factor_layout.setContentsMargins(0, 0, 0, 0)
        resize_factor_layout.addWidget(QLabel("Output resolution:"))
        self.resize_factor = QSpinBox(self)
        self.resize_factor.setSuffix(" %")
        self.resize_factor.setRange(10, 100)
        self.resize_factor.setValue(100)
        self.resize_factor.setSingleStep(10)
        self.resize_factor.valueChanged.connect(self.resize_factor_changed)
        resize_factor_layout.addWidget(self.resize_factor)

        self.interval = LabelRangeSlider(0, 1, self)
        self.interval.single_value_changed.connect(self.video_player.setCurrentFrame)

        self.generate_video_button = QPushButton(self)
        self.generate_video_button.setIcon(QIcon.fromTheme("document-save"))
        self.generate_video_button.setText("Generate and save video")
        self.generate_video_button.setToolTip(
            "Generate the video with the current settings"
        )
        self.generate_video_button.clicked.connect(self.generate_video)

        layout = self.centralWidget().layout()
        assert layout is not None

        self.controls_widget = QWidget(self)
        controls = QVBoxLayout()
        self.controls_widget.setLayout(controls)
        controls.addWidget(video_kind_row, alignment=Qt.AlignmentFlag.AlignHCenter)
        controls.addWidget(QHLine())
        identity_labels_switch = QToggleSwitch("Draw identity labels")
        identity_labels_switch.setChecked(True)
        identity_labels_switch.toggled.connect(self.id_labels.setEnabled)
        identity_labels_switch.toggled.connect(self.video_player.update)
        controls.addWidget(identity_labels_switch)
        controls.addWidget(self.id_labels)
        controls.addWidget(QHLine())
        controls.addWidget(self.general_trace_length_row)
        controls.addWidget(self.individual_size_row)
        controls.addWidget(self.resize_factor_row)
        controls.addWidget(render_in_gray_row)
        controls.addWidget(QHLine())
        controls.addWidget(QLabel("Video generation time-range:"))
        controls.addWidget(self.interval)
        controls.addWidget(self.generate_video_button)

        left_widget = QWidget(self)
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        left_widget.setLayout(left_layout)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.open_button)
        left_layout.addWidget(self.controls_widget)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(self.video_player)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        layout.addWidget(main_splitter)

        self.video_player.setEnabled(False)
        self.controls_widget.setEnabled(False)

    def closeEvent(self, event):
        if (
            hasattr(self, "session")
            and self.session is not None
            and self.id_labels.isEnabled()
        ):
            self.session.identities_labels = self.id_labels.get_labels()[1:]
            self.session.save()
        super().closeEvent(event)

    def generate_video(self):
        if self.individual_kind_switch.isChecked():
            output_path = QFileDialog.getExistingDirectory(
                self, "Select Output Folder", str(self.session.session_folder)
            )
        else:
            output_path = QFileDialog.getSaveFileName(
                self,
                "Save Video",
                str(
                    self.session.session_folder
                    / (self.session.video_paths[0].stem + "_tracked.avi")
                ),
                "Video Files (*.avi *.mp4)",
            )[0]

        if not output_path:
            return

        progress_dialog = QProgressDialog(
            "Generating Video", "Cancel", 0, 100, self, minimumDuration=0
        )

        def progress_callback(progress: float):
            progress_dialog.setValue(int(progress * 100))
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                raise KeyboardInterrupt("Video generation cancelled by user.")

        try:
            if self.individual_kind_switch.isChecked():
                generate_individual_video(
                    self.session,
                    self.session.trajectories_folder,
                    starting_frame=self.interval.value()[0],
                    ending_frame=self.interval.value()[1],
                    miniframe_size=self.individual_size.value(),
                    labels=(
                        self.id_labels.get_labels()[1:]
                        if self.id_labels.isEnabled()
                        else None
                    ),
                    callback=progress_callback,
                    draw_in_gray=not self.render_in_color.isChecked(),
                )
            else:
                generate_general_video(
                    self.session,
                    self.session.trajectories_folder,
                    starting_frame=self.interval.value()[0],
                    ending_frame=self.interval.value()[1],
                    centroid_trace_length=self.general_trace_length.value(),
                    resize_factor=self.resize_factor.value() / 100,
                    labels=(
                        self.id_labels.get_labels()[1:]
                        if self.id_labels.isEnabled()
                        else None
                    ),
                    callback=progress_callback,
                    output_path=output_path,
                    draw_in_gray=not self.render_in_color.isChecked(),
                )
        except KeyboardInterrupt:
            pass  # User cancelled the operation
        else:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("Done")
            msg_box.setText(f"Video generation finished, saved in {output_path}.")
            open_folder_btn = msg_box.addButton(
                "Open File Location", QMessageBox.ButtonRole.ActionRole
            )
            open_folder_btn.setIcon(QIcon.fromTheme("folder-open"))
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()

            if msg_box.clickedButton() == open_folder_btn:
                if Path(output_path).is_file():
                    # If output_path is a file, get its directory
                    output_path = str(Path(output_path).parent)
                try:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(output_path))
                except Exception as e:
                    QMessageBox.warning(
                        self, "Error Opening Folder", f"Could not open the folder: {e}"
                    )
        progress_dialog.close()

    def individual_size_changed(self, value: int):
        self.positions, self.individual_output_shape = get_geometries(
            self.session.n_animals, value
        )

        if not self.individual_kind_switch.isChecked():
            return
        self.video_player.canvas.adjust_zoom_to(*self.individual_output_shape[::-1])
        self.video_player.update()

    def toggle_video_kind(self, checked: bool):
        if checked:
            self.video_player.canvas.adjust_zoom_to(*self.individual_output_shape[::-1])
            self.individual_size_row.setVisible(True)
            self.general_trace_length_row.setVisible(False)
            self.resize_factor_row.setVisible(False)
        else:
            self.video_player.canvas.adjust_zoom_to(
                self.session.width, self.session.height
            )
            self.individual_size_row.setVisible(False)
            self.general_trace_length_row.setVisible(True)
            self.resize_factor_row.setVisible(True)

        self.video_player.update()

    def resize_factor_changed(self, value: int):
        self.resized_trajectories = (self.trajectories * value / 100).astype(int)
        self.video_player.update()

    def manageDropedPaths(self, paths: Sequence[str]) -> None:
        if not paths:
            return
        if len(paths) > 1:
            QMessageBox.warning(
                self,
                "Error with dragged files",
                "Cannot open more than one session at once",
            )
            return
        self.open(paths[0])

    def open(self, session_path: Path | str | None = None):
        session = open_session(self, self.settings, session_path)
        if session is None:
            return
        self.session = session

        try:
            self.trajectories = np.nan_to_num(
                load_trajectories(self.session.session_folder)["trajectories"], nan=-100
            )
        except FileNotFoundError as exc:
            QMessageBox.warning(
                self,
                "Trajectories not found",
                f"{exc}. Did the session finish tracking successfully?",
            )
            return

        self.resize_factor_changed(self.resize_factor.value())

        self.individual_size.setValue(2 * (int(session.median_body_length) // 2))

        self.video_player.update_video_paths(
            session.video_paths,
            session.number_of_frames,
            (session.width, session.height),
            session.frames_per_second,
        )

        self.positions, self.individual_output_shape = get_geometries(
            session.n_animals, self.individual_size.value()
        )

        self.interval.blockSignals(True)
        self.interval.setMaximum(session.number_of_frames - 1)
        self.interval.setValue((0, session.number_of_frames - 1))
        self.interval.blockSignals(False)

        self.id_labels.load_labels(
            session.identities_labels or list(map(str, range(1, session.n_animals + 1)))
        )

        self.cmap = get_cmap(session.n_animals)
        self.toggle_video_kind(self.individual_kind_switch.isChecked())

        self.video_player.setEnabled(True)
        self.controls_widget.setEnabled(True)

    def preview_frame(
        self,
        painter: CanvasPainter,
        frame_index: int,
        original_frame: np.ndarray | None,
    ):
        if original_frame is None or not hasattr(self, "session"):
            return

        assert original_frame.shape[2] == 3
        if not self.render_in_color.isChecked():
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        if self.individual_kind_switch.isChecked():
            miniframes = get_individual_miniframes(
                original_frame,
                (self.trajectories[frame_index]).astype(int),
                self.individual_size.value(),
            )

            frame = draw_collage_frame(
                self.positions,
                miniframes,
                self.individual_output_shape,
                self.id_labels.get_labels()[1:] if self.id_labels.isEnabled() else None,
            )
            preview_shape = self.video_player.video_drawing_origin
        else:
            resize_factor = self.resize_factor.value() / 100
            if resize_factor != 1:
                original_frame = cv2.resize(
                    original_frame, None, fx=resize_factor, fy=resize_factor
                )
            frame = draw_general_frame(
                original_frame,
                frame_index,
                self.resized_trajectories,
                self.general_trace_length.value(),
                self.cmap,
                self.id_labels.get_labels()[1:] if self.id_labels.isEnabled() else None,
            )
            preview_shape = QRect(0, 0, self.session.width, self.session.height)

        painter.drawPixmap(
            preview_shape,
            QPixmap.fromImage(
                QImage(
                    frame.data,
                    frame.shape[1],
                    frame.shape[0],
                    frame.shape[1] * 3,
                    QImage.Format.Format_BGR888,
                )
            ),
        )
