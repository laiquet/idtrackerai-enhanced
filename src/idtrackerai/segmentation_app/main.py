import logging
from pathlib import Path
from typing import Any

import toml
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from idtrackerai import Session
from idtrackerai.GUI_tools import GUIBase, QHLine, VideoPlayer
from idtrackerai.utils import pprint_dict

from .widgets import (
    AreaThresholds,
    BkgWidget,
    BlobInfoWidget,
    FrameAnalyzer,
    IntensityThresholds,
    OpenVideoWidget,
    ROIWidget,
    TrackingIntervalsWidget,
)


class SegmentationGUI(GUIBase):
    """The main class implementing the Segmentation app.
    From here, all widgets are coordinated and connected.

    Parameters
    ----------
    GUIBase : QMainWindow
        Base configuration for all idtracker.ai GUIs
    """

    def __init__(self, session: Session | None = None, signals: dict | None = None):
        super().__init__()

        self.setWindowTitle("Segmentation App")
        self.signals = signals or {}
        self.session = session or Session()
        self.documentation_url = (
            "https://idtracker.ai/latest/user_guide/segmentation_app.html"
        )

        self.open_widget = OpenVideoWidget(self, self.session.frames_per_episode)
        self.videoPlayer = VideoPlayer(self)
        self.frame_analyzer = FrameAnalyzer()
        self.blobInfo = BlobInfoWidget()
        self.bkg_widget = BkgWidget()
        self.ROI_Widget = ROIWidget(self)
        self.tracking_interval = TrackingIntervalsWidget(self)
        self.widgets_to_close.append(self.videoPlayer)

        self.intensity_thresholds = IntensityThresholds(self, min=0, max=255)
        self.area_thresholds = AreaThresholds()

        self.save_parameters = QPushButton("Save parameters")
        self.save_parameters.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.save_parameters.setShortcut("Ctrl+S")

        self.check_segm = QCheckBox("Stop tracking if #blobs > #animals")
        self.track_wo_id = QCheckBox("Track without identities")

        res_reduct_row = QHBoxLayout()
        resreduct_label = QLabel("Resolution")
        res_reduct_row.addWidget(resreduct_label)
        self.resreduct = QSpinBox()
        self.resreduct.setMaximum(100)
        self.resreduct.setMinimum(10)
        self.resreduct.setSingleStep(10)
        self.resreduct.setSuffix("%")
        res_reduct_row.addWidget(self.resreduct)
        res_reduct_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        n_animals_row = QHBoxLayout()
        n_animals_label = QLabel("Number of animals")
        n_animals_row.addWidget(n_animals_label)
        self.n_animals = QSpinBox()
        self.n_animals.setMaximum(1000)
        self.n_animals.setMinimum(0)
        self.n_animals.setSpecialValueText("unknown")
        n_animals_row.addWidget(self.n_animals)
        n_animals_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        session_row = QHBoxLayout()
        session_label = QLabel("Session")
        session_row.addWidget(session_label)
        self.session_name = SessionName()
        session_row.addWidget(self.session_name)
        session_row.addWidget(self.save_parameters)

        self.close_and_track_btn = QPushButton("Close window and track video")
        self.close_and_track_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Connecting widgets
        self.open_widget.path_clicked.connect(self.videoPlayer.setCurrentFrame)
        self.open_widget.new_video_paths.connect(self.new_video_paths)
        self.open_widget.new_episodes.connect(self.bkg_widget.set_new_video_paths)
        self.open_widget.new_parameters.connect(self.new_parameters)
        self.open_widget.video_paths_reordered.connect(
            self.videoPlayer.reorder_video_paths
        )
        self.open_widget.video_paths_reordered.connect(
            lambda paths: self.session_name.setPlaceholderText(
                "&".join(Path(path).stem for path in paths)
            )
        )
        self.resreduct.editingFinished.connect(self.resreduct.clearFocus)
        self.resreduct.valueChanged.connect(
            lambda x: self.ROI_Widget.set_resolution_reduction(x / 100)
        )
        self.resreduct.valueChanged.connect(
            lambda x: self.videoPlayer.set_resolution_reduction(x / 100)
        )
        self.resreduct.valueChanged.connect(
            lambda x: self.frame_analyzer.set_resolution_reduction(x / 100)
        )
        self.n_animals.editingFinished.connect(self.n_animals.clearFocus)
        self.n_animals.valueChanged.connect(self.blobInfo.setNAnimals)
        self.tracking_interval.newValue.connect(self.open_widget.set_tracking_interval)
        self.tracking_interval.newValue.connect(self.blobInfo.setTrackingIntervals)
        self.intensity_thresholds.newValue.connect(
            self.frame_analyzer.set_intensity_ths
        )
        self.session_name.editingFinished.connect(self.session_name.clearFocus)
        self.save_parameters.clicked.connect(self.save_parameters_func)
        self.area_thresholds.valueChanged.connect(self.frame_analyzer.set_area_ths)
        self.close_and_track_btn.clicked.connect(self.close_and_track_video)
        self.ROI_Widget.valueChanged.connect(self.frame_analyzer.set_ROI_mask)
        self.ROI_Widget.needToDraw.connect(self.videoPlayer.update)
        self.bkg_widget.new_bkg_data.connect(self.frame_analyzer.set_bkg)
        self.bkg_widget.new_bkg_data.connect(self.intensity_thresholds.bkg_changed)
        self.frame_analyzer.new_areas.connect(self.blobInfo.setAreas)
        self.frame_analyzer.new_parameters.connect(self.videoPlayer.update)
        self.videoPlayer.painting_time.connect(self.frame_analyzer.paint_on_canvas)
        self.videoPlayer.painting_time.connect(self.ROI_Widget.paint_on_canvas)
        self.videoPlayer.canvas.click_event.connect(self.ROI_Widget.click_event)

        # Tooltips texts
        tooltips = toml.load(Path(__file__).parent / "tooltips.toml")

        self.open_widget.button_open.setToolTip(tooltips["open_btn"])
        self.open_widget.single_file_label.setToolTip(tooltips["open_path_label"])
        self.open_widget.list_of_files.setToolTip(tooltips["open_path_list"])
        self.tracking_interval.setToolTip(tooltips["tacking_interval"])
        self.ROI_Widget.setToolTip(tooltips["region_of_interest"])
        self.ROI_Widget.exclusive_rois.setToolTip(tooltips["exclusive_rois"])
        self.bkg_widget.setToolTip(tooltips["background_subtraction"])
        self.bkg_widget.bkg_stat.setToolTip(tooltips["background_stat"])
        self.bkg_widget.view_bkg.setToolTip(tooltips["background_view"])
        self.n_animals.setToolTip(tooltips["number_of_animals"])
        n_animals_label.setToolTip(tooltips["number_of_animals"])
        self.check_segm.setToolTip(tooltips["check_segm"])
        self.area_thresholds.setToolTip(tooltips["area_thresholds"])
        self.resreduct.setToolTip(tooltips["resolution_reduction"])
        resreduct_label.setToolTip(tooltips["resolution_reduction"])
        self.track_wo_id.setToolTip(tooltips["track_wo_id"])
        self.save_parameters.setToolTip(tooltips["save_params"])
        self.close_and_track_btn.setToolTip(tooltips["close_and_track"])
        self.blobInfo.setToolTip(tooltips["blobs_info"])
        self.session_name.setToolTip(tooltips["session_name"])
        session_label.setToolTip(tooltips["session_name"])
        self.intensity_thresholds.setToolTips(
            tooltips["intensity_thresholds_nobkg"],
            tooltips["intensity_thresholds_yesbkg"],
        )

        left_layout = QVBoxLayout()
        self.open_widget.layout().setContentsMargins(0, 8, 0, 0)
        left_layout.addWidget(self.open_widget)
        for widget in (
            QHLine(),
            res_reduct_row,
            self.tracking_interval,
            self.ROI_Widget,
            QHLine(),
            n_animals_row,
            self.bkg_widget,
            self.intensity_thresholds,
            self.area_thresholds,
            QHLine(),
            self.check_segm,
            self.track_wo_id,
        ):
            if isinstance(widget, (QVBoxLayout, QHBoxLayout)):
                widget.setContentsMargins(0, 0, 0, 0)
                superwidget = QWidget()
                superwidget.setLayout(widget)
                left_layout.addWidget(
                    superwidget, alignment=Qt.AlignmentFlag.AlignVCenter
                )
            else:
                lay = widget.layout()
                if lay is not None:
                    lay.setContentsMargins(0, 0, 0, 0)
                left_layout.addWidget(
                    widget,
                    alignment=Qt.AlignmentFlag.AlignVCenter,
                    stretch=1 if isinstance(widget, QHLine) else 3,
                )
        left_layout.addLayout(session_row)
        left_layout.addWidget(self.close_and_track_btn)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.addWidget(self.blobInfo)
        self.videoPlayer.layout().setContentsMargins(8, 8, 0, 0)
        self.right_splitter.addWidget(self.videoPlayer)
        self.right_splitter.setSizes([200, 600])

        left = QWidget()
        left.setLayout(left_layout)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left)
        main_splitter.addWidget(self.right_splitter)
        main_splitter.setSizes([400, 600])
        self.centralWidget().layout().addWidget(main_splitter)
        self.list_of_widgets = self.get_list_of_widgets(left_layout)
        for widget in self.list_of_widgets:
            widget.setEnabled(False)
        self.right_splitter.setEnabled(False)
        self.enabled = False
        self.open_widget.setEnabled(True)

        self.setTabOrder(self.resreduct, self.videoPlayer.canvas)
        self.setTabOrder(self.videoPlayer.canvas, self.resreduct)
        for widget in self.findChildren(QCheckBox):
            assert isinstance(widget, QWidget)
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        QTimer.singleShot(0, self.load_parameters)

    def load_parameters(self):
        """Sets all widgets to the values indicated by self.session"""
        self.open_widget.open_video_paths(self.session.video_paths)
        self.resreduct.setValue(int(self.session.resolution_reduction * 100))
        self.tracking_interval.setValue(self.session.tracking_intervals)
        self.ROI_Widget.setValue(self.session.roi_list, self.session.exclusive_rois)
        self.intensity_thresholds.setValue(self.session.intensity_ths)
        self.area_thresholds.setValue(self.session.area_ths)
        self.n_animals.setValue(self.session.number_of_animals)
        self.track_wo_id.setChecked(self.session.track_wo_identities)
        self.check_segm.setChecked(self.session.check_segmentation)
        self.session_name.setText(self.session.name)
        self.bkg_widget.bkg_stat.setCurrentText(
            self.session.background_subtraction_stat.capitalize()
        )
        self.bkg_widget.bkg_thread.n_frames_for_background = (
            self.session.number_of_frames_for_background
        )
        self.bkg_widget.checkBox.setChecked(self.session.use_bkg)
        self.videoPlayer.update()

    def new_parameters(self, params: dict):
        "Receives new toml parameters from OpenVideoWidget"
        unrecognized_params = self.session.set_parameters(**params, reset=True)
        if unrecognized_params:
            QMessageBox.warning(
                self,
                "Error loading parameters",
                f"Unrecognized parameters in toml file:\n\n{unrecognized_params}",
            )
        self.load_parameters()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.ROI_Widget.add.setChecked(False)

    def close_and_track_video(self):
        """Action when clicked "close and track video".
        It gathers widgets parameters, writes them in self.user_params and exits"""
        parameters = self.out_parameters()
        if self.unacceptable_parameters(parameters):
            return

        logging.info(pprint_dict(parameters, "GUI params"), extra={"markup": True})
        self.session.set_parameters(**parameters)
        self.session.background_from_segmentation_gui = self.bkg_widget.getBkg()
        # signal to start tracking after closing app
        self.signals["run_idtrackerai"] = True
        self.close()

    def getSessionName(self) -> str:
        return (
            self.session_name.text() or self.session_name.placeholderText() or "no_name"
        )

    def out_parameters(self) -> dict:
        """Generates dict of all widgets content

        Returns
        -------
        dict
            Parameter dict containing all widgets content
        """
        out: dict[str, Any] = {
            "video_paths": self.open_widget.getVideoPaths(),
            "intensity_ths": self.intensity_thresholds.value(),
            "area_ths": self.area_thresholds.value(),
            "tracking_intervals": self.tracking_interval.value(),
            "number_of_animals": self.n_animals.value(),
            "use_bkg": self.bkg_widget.checkBox.isChecked(),
            "check_segmentation": self.check_segm.isChecked(),
            "resolution_reduction": self.resreduct.value() / 100,
            "track_wo_identities": self.track_wo_id.isChecked(),
            "roi_list": self.ROI_Widget.getValue(),
        }

        if self.session_name.text():  # put the name at the first position
            out = {"name": self.session_name.text()} | out

        if self.bkg_widget.checkBox.isChecked():
            out["background_subtraction_stat"] = (
                self.bkg_widget.bkg_stat.currentText().lower()
            )

        if (
            self.ROI_Widget.exclusive_rois.isVisible()
            and self.ROI_Widget.exclusive_rois.isEnabled()
        ):
            out["exclusive_rois"] = self.ROI_Widget.exclusive_rois.isChecked()
        return out

    def unacceptable_parameters(self, parameters: dict) -> bool:
        if (
            parameters["number_of_animals"] == 0
            and not parameters["track_wo_identities"]
        ):
            QMessageBox.warning(
                self,
                "Missing parameters",
                'Please, define the number of animals in the video or check "Track'
                ' without identities".\n\nEven if tracking without identities,'
                " adding the number of animals is recommended to improve the"
                " individual/crossing blob detection.",
            )
            return True
        if parameters["roi_list"] is not None and len(parameters["roi_list"]) == 0:
            QMessageBox.warning(
                self,
                "Missing parameters",
                "Please, add a region of interest or uncheck the Regions of"
                " interest parameter.",
            )
            return True
        return False

    def save_parameters_func(self):
        parameters = self.out_parameters()

        if self.unacceptable_parameters(parameters):
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameter file",
            str(Path.cwd() / (self.getSessionName() + ".toml")),
            filter="TOML (*.toml)",
        )
        if not fileName:
            return

        with open(fileName, "w", encoding="utf_8") as file:
            for key, value in parameters.items():
                file.write(f"{key} = {toml_format(value)}\n")

    def new_video_paths(
        self,
        video_paths: list[str],
        video_size: tuple[int, int],
        n_frames: int,
        fps: int,
        episodes: list,
    ):
        self.session_name.setPlaceholderText(
            "&".join(Path(path).stem for path in video_paths)
        )
        self.ROI_Widget.set_video_size(video_size)
        self.videoPlayer.setEnabled(False)
        self.tracking_interval.reset(n_frames)
        self.frame_analyzer.drawn_frame = -1
        self.bkg_widget.set_new_video_paths(video_paths, episodes)
        self.ROI_Widget.list.ListChanged.emit()
        self.videoPlayer.update_video_paths(video_paths, n_frames, video_size, fps)

        if not self.enabled:
            for widget in self.list_of_widgets:
                widget.setEnabled(True)
            self.enabled = True
            self.right_splitter.setEnabled(True)

        self.videoPlayer.setEnabled(True)
        self.videoPlayer.update()


def toml_format(value: Any, width: int = 50) -> str:
    """Custom .toml formatter.

    Parameters
    ----------
    value : Any
        The value to format
    width : int, optional
        Maximum line width before wrapping, by default 50

    Returns
    -------
    str
        Formatted value
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return repr(value)
    if value is None:
        return '""'
    if not value:
        return "[]"
    value = list(value)

    if len(repr(value)) < width:
        return repr(value)

    s = "[\n"
    for item in value:
        s += f"    {repr(item)},\n"
    s += "]"
    return s


class SessionName(QLineEdit):
    "Double click to set the placeholder text as the text"

    def mouseDoubleClickEvent(self, event):
        if not self.text() and self.placeholderText():
            self.setText(self.placeholderText())
        else:
            super().mouseDoubleClickEvent(event)
