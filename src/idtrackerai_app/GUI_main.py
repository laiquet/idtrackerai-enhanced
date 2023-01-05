from importlib.resources import files
from pathlib import Path

import toml
from idtrackerai_app.GUI_Widgets import (
    BkgWidget,
    BlobInfoWidget,
    FrameAnalyzer,
    OpenVideoWidget,
    ROIWidget,
    SetupPointsWidget,
    TrackingIntervalsWidget,
    VideoPlayer,
)
from idtrackerai_app.widgets_utils import (
    GUIBase,
    LabelRangeSlider,
    WrappedLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from idtrackerai.utils import conf


class SegmentationGUI(GUIBase):
    def __init__(self, GUI_out_params: dict):
        super().__init__()

        self.setWindowTitle("idTracker.ai | segmentation GUI")
        self.GUI_out_params = GUI_out_params

        self.open_widget = OpenVideoWidget(self)
        self.VideoPlayer = VideoPlayer()
        self.frame_analyzer = FrameAnalyzer(self, self.VideoPlayer.canvas.ax)
        self.BlobInfo = BlobInfoWidget(self)
        self.bkg_widget = BkgWidget(self)
        self.setup_widget = SetupPointsWidget(self, self.VideoPlayer.canvas.ax)
        self.ROI_Widget = ROIWidget(self, self.VideoPlayer.canvas.ax)
        self.tracking_interval = TrackingIntervalsWidget(parent=self)

        self.resreduct = QSpinBox(
            maximum=100,
            minimum=10,
            singleStep=10,
            suffix="%",
            value=int(conf.RES_REDUCTION_DEFAULT * 100),
        )

        self.check_segm = QCheckBox("Check segmentation")

        self.n_animals = QSpinBox(maximum=100, minimum=1)

        self.intensity_thresholds = LabelRangeSlider(
            min=conf.MIN_THRESHOLD, max=conf.MAX_THRESHOLD
        )

        self.area_thresholds = LabelRangeSlider(
            min=conf.AREA_LOWER, max=conf.AREA_UPPER
        )

        self.session = QLineEdit()
        self.session.setPlaceholderText("Example: text, experiment_32A, ...")
        self.session.setFixedHeight(28)

        self.save_parameters = QPushButton("Save parameters")
        self.save_parameters.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.track_wo_id = QCheckBox("Track without identities")

        res_reduct_row = QHBoxLayout()
        res_reduct_row.addWidget(QLabel("Resolution reduction"))
        res_reduct_row.addWidget(self.resreduct)

        n_animals_row = QHBoxLayout()
        n_animals_row.addWidget(WrappedLabel("Number of animals"))
        n_animals_row.addWidget(self.n_animals)
        n_animals_row.addWidget(self.check_segm)

        intensity_row = QHBoxLayout()
        intensity_row.addWidget(QLabel("Intensity thresholds"))
        intensity_row.addWidget(self.intensity_thresholds)

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
        # TODO clean possible redundant connections
        self.open_widget.pause_video.connect(self.VideoPlayer.stop_all)
        self.open_widget.path_clicked.connect(self.VideoPlayer.setCurrentFrame)
        self.open_widget.new_video_paths.connect(self.new_video_paths)
        self.open_widget.video_paths_reordered.connect(
            self.VideoPlayer.reorder_video_paths
        )
        self.resreduct.editingFinished.connect(self.resreduct.clearFocus)
        self.resreduct.valueChanged.connect(
            self.frame_analyzer.set_resolution_reduction
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
        self.area_thresholds.newValue.connect(self.frame_analyzer.set_area_ths)
        self.track_btn.clicked.connect(self.close_and_track_video)
        self.ROI_Widget.valueChanged.connect(self.frame_analyzer.set_ROI_mask)
        self.ROI_Widget.needToDraw.connect(self.VideoPlayer.update_player)
        self.ROI_Widget.valueChanged.connect(self.bkg_widget.set_ROI)
        self.bkg_widget.new_bkg_data.connect(self.frame_analyzer.set_bkg)
        self.setup_widget.needToDraw.connect(self.VideoPlayer.update_player)
        self.frame_analyzer.new_areas.connect(self.BlobInfo.setAreas)
        self.frame_analyzer.new_parameters.connect(self.VideoPlayer.update_player)
        self.VideoPlayer.blit_event.connect(self.frame_analyzer.draw_artists)
        self.VideoPlayer.blit_event.connect(self.ROI_Widget.draw_artists)
        self.VideoPlayer.blit_event.connect(self.setup_widget.draw_artists)
        self.VideoPlayer.canvas.click_event.connect(self.ROI_Widget.click_event)
        self.VideoPlayer.canvas.click_event.connect(self.setup_widget.click_event)
        self.VideoPlayer.canvas.click_event.connect(self.clearFocus)

        # Tooltips texts
        tooltips = toml.load(files("idtrackerai_app") / "tooltips.toml")
        self.check_segm.setToolTip(tooltips["check_segm"])
        self.area_thresholds.setToolTip(tooltips["area_thresholds"])
        self.intensity_thresholds.setToolTip(tooltips["intensity_thresholds"])

        # Define widget structure

        main_layout = QHBoxLayout()
        self.centralWidget().setLayout(main_layout)
        left = QVBoxLayout()
        right = QVBoxLayout()
        main_layout.addLayout(left, 40)
        main_layout.addLayout(right, 60)
        left.addLayout(self.open_widget)
        left.addLayout(self.tracking_interval)
        left.addLayout(self.ROI_Widget, 0)
        left.addLayout(self.bkg_widget)
        left.addLayout(res_reduct_row)
        left.addLayout(n_animals_row)
        left.addLayout(intensity_row)
        left.addLayout(area_row)
        left.addLayout(self.setup_widget)
        left.addWidget(self.track_wo_id)
        left.addLayout(session_row)
        left.addWidget(self.track_btn)
        right.addLayout(self.BlobInfo, 30)
        right.addWidget(self.VideoPlayer, 70)

        self.list_of_widgets = self.get_list_of_widgets(main_layout)
        for widget in self.list_of_widgets:
            widget.setEnabled(False)
        self.enabled = False
        self.open_widget.button_open.setEnabled(True)

        self.load_parameters(self.GUI_out_params)

        self.setTabOrder(self.tracking_interval.multiple_text, self.VideoPlayer.canvas)
        self.setTabOrder(self.VideoPlayer.canvas, self.tracking_interval.multiple_text)
        self.setTabOrder(self.VideoPlayer.canvas, self.ROI_Widget.add)
        self.setTabOrder(self.VideoPlayer.canvas, self.resreduct)
        for widget in self.findChildren(QCheckBox):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.center_window()

    def load_parameters(self, load_dict: dict):
        self.open_widget.open_video_paths(
            video_paths=load_dict.get("video_paths", None)
        )

        resolution_reduction = load_dict.get("resolution_reduction", 1)
        self.resreduct.setValue(int(resolution_reduction * 100))

        self.tracking_interval.setValue(load_dict.get("tracking_intervals", None))

        self.setup_widget.setValue(load_dict.get("setup_points", None))
        self.ROI_Widget.setValue(load_dict.get("ROI_list", None))

        self.intensity_thresholds.setValue(
            load_dict.get(
                "intensity_ths",
                (conf.MIN_THRESHOLD_DEFAULT, conf.MAX_THRESHOLD_DEFAULT),
            )
        )

        self.area_thresholds.setValue(
            load_dict.get("areas_ths", (conf.MIN_AREA_DEFAULT, conf.MAX_AREA_DEFAULT))
        )

        self.n_animals.setValue(
            load_dict.get("number_of_animals", conf.NUMBER_OF_ANIMALS_DEFAULT)
        )

        self.track_wo_id.setChecked(load_dict.get("track_wo_identities", False))

        self.check_segm.setChecked(load_dict.get("check_segmentation", False))
        self.session.setText(load_dict.get("session", ""))

        if load_dict.get("use_bkg", False):
            self.bkg_widget.CheckBox.click()

        if self.enabled:
            self.VideoPlayer.update_player()

    def close_and_track_video(self):
        self.GUI_out_params.update(self.out_parameters())
        self.GUI_out_params["bkg_model"] = self.bkg_widget.getBkg()
        # signal to start tracking after closing app
        self.GUI_out_params["run_idtrackerai"] = True
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
            "use_bkg": self.bkg_widget.CheckBox.isChecked(),
            "check_segmentation": self.check_segm.isChecked(),
            "resolution_reduction": self.resreduct.value() / 100,
            "track_wo_identities": self.track_wo_id.isChecked(),
            "ROI_list": self.ROI_Widget.getValue(),
            "setup_points": self.setup_widget.getValue(),
        }

    def save_parameters_func(self):

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
        if key in (Qt.Key_Enter, Qt.Key_Return):
            self.ROI_Widget.enter_key_event()
            self.setup_widget.enter_key_event()
        else:
            self.VideoPlayer.redirect_keyPressEvent(key)

    def processed_keyReleaseEvent(self, key: int):
        self.VideoPlayer.redirect_keyReleaseEvent(key)

    def new_video_paths(self, video_paths, video_size, n_frames, fps, episodes):
        # FIXME
        self.ROI_Widget.set_video_size(video_size)
        self.VideoPlayer.setEnabled(False)
        self.tracking_interval.reset(n_frames)
        self.BlobInfo.bg = None
        self.frame_analyzer.drawn_frame = -1
        self.bkg_widget.set_new_video_paths(video_paths, episodes)
        self.ROI_Widget.ListChanged.emit()
        self.VideoPlayer.update_video_paths(
            video_paths,
            n_frames,
            video_size,
            fps,
        )

        if not self.enabled:
            for widget in self.list_of_widgets:
                widget.setEnabled(True)
            self.enabled = True

        self.VideoPlayer.setEnabled(True)
        # self.bkg_widget.reset()
        self.VideoPlayer.update_player()


def toml_format(value: list[str] | bool, width=50) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float, str)):
        return repr(value)

    if not value:
        return "[]"

    if len(repr(value)) < width:
        return repr(value)

    s = "[\n"
    for item in value:
        s += f"    {repr(item)},\n"
    s += "]"
    return s
