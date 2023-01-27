import logging
from pathlib import Path

import toml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.utils import pprint_dict

from .GUI_Widgets import (
    BkgWidget,
    BlobInfoWidget,
    FrameAnalyzer,
    IntensityThresholds,
    OpenVideoWidget,
    ROIWidget,
    SetupPointsWidget,
    TrackingIntervalsWidget,
    VideoPlayer,
)
from .widgets_utils import (
    GUIBase,
    LabelRangeSlider,
    MessageBox,
    WrappedLabel,
    WrappedCheckBox,
)


class SegmentationGUI(GUIBase):
    def __init__(self, GUI_out_params: dict):
        super().__init__()

        self.setWindowTitle("idTracker.ai | segmentation GUI")
        self.user_params = GUI_out_params

        self.warning_dialog = MessageBox(self, "Warning", "warning")

        self.open_widget = OpenVideoWidget(self)
        self.videoPlayer = VideoPlayer(self)
        self.frame_analyzer = FrameAnalyzer(self)
        self.BlobInfo = BlobInfoWidget(self)
        self.bkg_widget = BkgWidget(self)
        self.setup_widget = SetupPointsWidget(self)
        self.ROI_Widget = ROIWidget(self)
        self.tracking_interval = TrackingIntervalsWidget(self)

        self.intensity_thresholds = IntensityThresholds(self, min=0, max=255)
        self.area_thresholds = LabelRangeSlider(
            parent=self, min=0, max=60000, block_upper=False
        )

        self.session = QLineEdit()
        self.session.setPlaceholderText("Example: test, experiment_32A, ...")
        self.session.setFixedHeight(28)

        self.save_parameters = QPushButton("Save parameters")
        self.save_parameters.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.track_wo_id = QCheckBox("Track without identities")

        res_reduct_row = QHBoxLayout()
        res_reduct_row.addWidget(QLabel("Resolution reduction"))
        self.resreduct = QSpinBox()
        self.resreduct.setMaximum(100)
        self.resreduct.setMinimum(10)
        self.resreduct.setSingleStep(10)
        self.resreduct.setSuffix("%")
        res_reduct_row.addWidget(self.resreduct)

        n_animals_row = QHBoxLayout()
        n_animals_row.addWidget(WrappedLabel("Number of animals"))
        self.n_animals = QSpinBox()
        self.n_animals.setMaximum(100)
        self.n_animals.setMinimum(0)
        n_animals_row.addWidget(self.n_animals)
        self.check_segm = WrappedCheckBox("Check segmentation")
        n_animals_row.addWidget(self.check_segm)

        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Area thresholds"))
        area_row.addWidget(self.area_thresholds)

        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("Session"))
        session_row.addWidget(self.session)
        session_row.addWidget(self.save_parameters)

        self.track_btn = QPushButton("Close window and track video")
        self.track_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Connecting widgets
        self.open_widget.pause_video.connect(self.videoPlayer.stop_all)
        self.open_widget.path_clicked.connect(self.videoPlayer.setCurrentFrame)
        self.open_widget.new_video_paths.connect(self.new_video_paths)
        self.open_widget.video_paths_reordered.connect(
            self.videoPlayer.reorder_video_paths
        )
        self.resreduct.editingFinished.connect(self.resreduct.clearFocus)
        self.resreduct.valueChanged.connect(
            lambda x: self.videoPlayer.set_resolution_reduction(x / 100)
        )
        self.resreduct.valueChanged.connect(
            lambda x: self.frame_analyzer.set_resolution_reduction(x / 100)
        )
        self.n_animals.editingFinished.connect(self.n_animals.clearFocus)
        self.n_animals.valueChanged.connect(self.BlobInfo.setNAnimals)
        self.open_widget.new_episodes.connect(self.bkg_widget.set_new_video_paths)
        self.tracking_interval.newValue.connect(self.open_widget.set_tracking_interval)
        self.tracking_interval.newValue.connect(self.BlobInfo.setTrackingIntervals)
        self.intensity_thresholds.newValue.connect(
            self.frame_analyzer.set_intensity_ths
        )
        self.session.editingFinished.connect(self.session.clearFocus)
        self.save_parameters.clicked.connect(self.save_parameters_func)
        self.area_thresholds.valueChanged.connect(self.frame_analyzer.set_area_ths)
        self.track_btn.clicked.connect(self.close_and_track_video)
        self.ROI_Widget.valueChanged.connect(self.frame_analyzer.set_ROI_mask)
        self.ROI_Widget.needToDraw.connect(self.videoPlayer.update)
        self.ROI_Widget.valueChanged.connect(self.bkg_widget.set_ROI)
        self.bkg_widget.new_bkg_data.connect(self.frame_analyzer.set_bkg)
        self.bkg_widget.new_bkg_data.connect(self.intensity_thresholds.bkg_changed)
        self.setup_widget.needToDraw.connect(self.videoPlayer.update)
        self.frame_analyzer.new_areas.connect(self.BlobInfo.setAreas)
        self.frame_analyzer.new_parameters.connect(self.videoPlayer.update)
        self.videoPlayer.painting_time.connect(self.frame_analyzer.paint_on_canvas)
        self.videoPlayer.painting_time.connect(self.ROI_Widget.paint_on_canvas)
        self.videoPlayer.painting_time.connect(self.setup_widget.paint_on_canvas)
        self.videoPlayer.canvas.click_event.connect(self.ROI_Widget.click_event)
        self.videoPlayer.canvas.click_event.connect(self.setup_widget.click_event)

        # Tooltips texts
        tooltips = toml.load(Path(__file__).parent / "tooltips.toml")
        self.check_segm.setToolTip(tooltips["check_segm"])
        self.area_thresholds.setToolTip(tooltips["area_thresholds"])
        self.intensity_thresholds.setToolTip(tooltips["intensity_thresholds"])

        # Define widget structure
        left_layout = QVBoxLayout()

        widgets = (
            self.open_widget,
            self.tracking_interval,
            self.ROI_Widget,
            self.bkg_widget,
            self.intensity_thresholds,
            n_animals_row,
            res_reduct_row,
            area_row,
            self.setup_widget,
            self.track_wo_id,
        )

        for widget in widgets:
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
                left_layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignVCenter)
        left_layout.addLayout(session_row)
        left_layout.addWidget(self.track_btn)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.addWidget(self.BlobInfo)
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
        self.center_window()

        self.setTabOrder(self.tracking_interval.multiple_text, self.videoPlayer.canvas)
        self.setTabOrder(self.videoPlayer.canvas, self.tracking_interval.multiple_text)
        self.setTabOrder(self.videoPlayer.canvas, self.ROI_Widget.add)
        self.setTabOrder(self.videoPlayer.canvas, self.resreduct)
        for widget in self.findChildren(QCheckBox):
            assert isinstance(widget, QWidget)
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        QTimer.singleShot(0, lambda: self.load_parameters(self.user_params))

    def load_parameters(self, load_dict: dict):
        self.open_widget.open_video_paths(load_dict.get("video_paths", None))
        self.resreduct.setValue(int(load_dict["resolution_reduction"] * 100))
        self.tracking_interval.setValue(load_dict["tracking_intervals"])
        self.setup_widget.setValue(load_dict["setup_points"])
        self.ROI_Widget.setValue(load_dict["roi_list"])
        self.intensity_thresholds.setValue(load_dict.get("intensity_ths", (0, 155)))
        self.area_thresholds.setValue(load_dict.get("areas_ths", (100, 99999999999)))
        self.n_animals.setValue(load_dict.get("number_of_animals", 0))
        self.track_wo_id.setChecked(load_dict["track_wo_identities"])
        self.check_segm.setChecked(load_dict["check_segmentation"])
        self.session.setText(load_dict.get("session", ""))
        self.bkg_widget.checkBox.setChecked(load_dict["use_bkg"])

        if self.enabled:
            self.videoPlayer.update()

    def close_and_track_video(self):
        if self.n_animals.value() == 0:
            self.warning_dialog.exec(
                "Please, define the number of animals in the video"
            )
            return
        GUI_params = self.out_parameters()
        logging.info(pprint_dict(GUI_params, "GUI params"), extra={"markup": True})
        self.user_params.update(GUI_params)
        self.user_params["bkg_model"] = self.bkg_widget.getBkg()
        # signal to start tracking after closing app
        self.user_params["run_idtrackerai"] = True
        self.close()

    def getSessionName(self) -> str:
        session_name = self.session.text()
        return session_name if session_name else "no_name"

    def out_parameters(self) -> dict:
        return {
            "session": self.getSessionName(),
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
            "setup_points": self.setup_widget.getValue(),
        }

    def save_parameters_func(self):
        if self.n_animals.value() == 0:
            self.warning_dialog.exec(
                "Please, define the number of animals in the video"
            )
            return
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameter file",
            str(Path.cwd() / (self.getSessionName() + ".toml")),
            filter="TOML (*.toml)",
        )
        if not fileName:
            return

        with open(fileName, "w") as file:
            for key, value in self.out_parameters().items():
                file.write(f"{key} = {toml_format(value)}\n")

    def processed_keyPressEvent(self, key: int):
        if key in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.ROI_Widget.enter_key_event()
            self.setup_widget.enter_key_event()
        else:
            self.videoPlayer.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.videoPlayer.redirect_keyReleaseEvent(key)

    def new_video_paths(self, video_paths, video_size, n_frames, fps, episodes):
        self.ROI_Widget.set_video_size(video_size)
        self.videoPlayer.setEnabled(False)
        self.tracking_interval.reset(n_frames)
        self.frame_analyzer.drawn_frame = -1
        self.bkg_widget.set_new_video_paths(video_paths, episodes)
        self.ROI_Widget.ListChanged.emit()
        self.videoPlayer.update_video_paths(video_paths, n_frames, video_size, fps)

        if not self.enabled:
            for widget in self.list_of_widgets:
                widget.setEnabled(True)
            self.enabled = True
            self.right_splitter.setEnabled(True)

        self.videoPlayer.setEnabled(True)
        self.videoPlayer.update()


def toml_format(value: list[str] | bool, width=50) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float, str)):
        return repr(value)
    elif value is None:
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
