import matplotlib.style as mplstyle
from PyQt6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

mplstyle.use("fast")
import logging
from pathlib import Path

from idtrackerai_app.GUI_Widgets import (
    BkgWidget,
    BlobInfoWidget,
    OpenVideoWidget,
    ROIWidget,
    SetupPointsWidget,
    TrackingIntervalsWidget,
    VideoPlayerWidget,
)
from idtrackerai_app.widgets_utils import LabelRangeSlider, WrappedLabel
from matplotlib.pyplot import rcParams
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QKeyEvent

from idtrackerai.utils import conf


class Window(QWidget):
    def __init__(self, GUI_out_params):

        logging.debug("Initializing GUI")
        super().__init__()

        # Clean all the default keyboard shortcuts of matplotlib
        for action, keybindings in rcParams.items():
            if action.startswith("keymap."):
                keybindings.clear()
        # rcParams["font.family"] = "sans-serif"
        # rcParams["font.sans-serif"] = "Arial"

        self.setWindowTitle("idTracker.ai | segmentation GUI")
        self.setGeometry(100, 60, 1000, 800)
        self.GUI_out_params = GUI_out_params
        self.param_funcs = {}

        self.open_widget = OpenVideoWidget(self)
        self.VideoPlayer = VideoPlayerWidget(self, self.param_funcs)
        self.BlobInfo = BlobInfoWidget(self)
        self.bkg_widget = BkgWidget(self, self.param_funcs)
        self.setup_widget = SetupPointsWidget(self, self.VideoPlayer.canvas.ax)
        self.ROI_Widget = ROIWidget(
            self, self.param_funcs, self.VideoPlayer.canvas.ax
        )
        self.tracking_interval = TrackingIntervalsWidget(parent=self)

        self.resreduct = QSpinBox(
            maximum=100,
            minimum=10,
            singleStep=10,
            suffix="%",
            value=int(conf.RES_REDUCTION_DEFAULT * 100),
        )

        self.check_segm = QCheckBox("Check segmentation")

        self.n_animals = QSpinBox(
            maximum=100,
            minimum=1,
        )

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
        self.open_widget.video_paths_reordered.connect(self.bkg_widget.reset)
        self.open_widget.video_paths_reordered.connect(
            self.VideoPlayer.reorder_video_paths
        )
        self.resreduct.editingFinished.connect(self.resreduct.clearFocus)
        self.resreduct.valueChanged.connect(self.VideoPlayer.update_player)
        self.n_animals.editingFinished.connect(self.n_animals.clearFocus)
        self.n_animals.valueChanged.connect(self.BlobInfo.setNAnimals)
        self.tracking_interval.newValue.connect(self.bkg_widget.reset)
        self.tracking_interval.newValue.connect(
            self.BlobInfo.setTrackingIntervals
        )
        self.intensity_thresholds.newValue.connect(
            self.VideoPlayer.update_player
        )
        self.session.editingFinished.connect(self.session.clearFocus)
        self.save_parameters.clicked.connect(self.save_parameters_func)
        self.area_thresholds.newValue.connect(self.VideoPlayer.update_player)
        self.track_btn.clicked.connect(self.close_and_track_video)
        self.ROI_Widget.update_player.connect(self.VideoPlayer.update_player)
        self.ROI_Widget.ListChanged.connect(self.bkg_widget.partial_reset)
        self.bkg_widget.new_bkg_data.connect(self.VideoPlayer.update_player)
        self.setup_widget.update_player.connect(self.VideoPlayer.update_player)
        self.VideoPlayer.new_areas.connect(self.BlobInfo.setAreas)
        self.VideoPlayer.blit_event.connect(self.ROI_Widget.draw_artists)
        self.VideoPlayer.blit_event.connect(self.setup_widget.draw_artists)
        self.VideoPlayer.canvas.click_event.connect(
            self.ROI_Widget.click_event
        )
        self.VideoPlayer.canvas.click_event.connect(
            self.setup_widget.click_event
        )
        self.VideoPlayer.canvas.click_event.connect(self.clearFocus)

        # Define widget structure
        main_layout = QHBoxLayout(self)
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
        self.build_param_funcs()

        self.list_of_widgets = self.get_list_of_widgets(main_layout)
        for widget in self.list_of_widgets:
            widget.setEnabled(False)
        self.enabled = False
        self.open_widget.button_open.setEnabled(True)

        self.load_parameters(self.GUI_out_params)

        self.setTabOrder(
            self.tracking_interval.multiple_text, self.VideoPlayer.canvas
        )
        self.setTabOrder(
            self.VideoPlayer.canvas, self.tracking_interval.multiple_text
        )
        for widget in self.findChildren(QCheckBox):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def load_parameters(self, load_dict: dict):
        self.open_widget.open_video_paths(
            video_paths=load_dict.get("video_paths", None)
        )

        resolution_reduction = load_dict.get("resolution_reduction", 1)
        self.resreduct.setValue(int(resolution_reduction * 100))

        self.tracking_interval.setValue(
            load_dict.get("tracking_intervals", None)
        )

        self.setup_widget.setValue(load_dict.get("setup_points", None))
        self.ROI_Widget.setValue(load_dict.get("ROI_list", None))

        self.intensity_thresholds.setValue(
            load_dict.get(
                "intensity_ths",
                (conf.MIN_THRESHOLD_DEFAULT, conf.MAX_THRESHOLD_DEFAULT),
            )
        )

        self.area_thresholds.setValue(
            load_dict.get(
                "areas_ths", (conf.MIN_AREA_DEFAULT, conf.MAX_AREA_DEFAULT)
            )
        )

        self.n_animals.setValue(
            load_dict.get("number_of_animals", conf.NUMBER_OF_ANIMALS_DEFAULT)
        )

        self.track_wo_id.setChecked(
            load_dict.get("track_wo_identities", False)
        )

        self.check_segm.setChecked(load_dict.get("check_segmentation", False))
        self.session.setText(load_dict.get("session", ""))

        if load_dict.get("use_bkg", False):
            self.bkg_widget.CheckBox.click()

        if self.enabled:
            self.VideoPlayer.update_player()

    def build_param_funcs(self):
        # TODO check if they are all used
        self.param_funcs["tracking_intervals"] = self.tracking_interval.value
        self.param_funcs["intensity_ths"] = self.intensity_thresholds.value
        self.param_funcs["area_ths"] = self.area_thresholds.value
        self.param_funcs["number_of_animals"] = self.n_animals.value
        self.param_funcs["resolution_reduction"] = (
            lambda: self.resreduct.value() / 100
        )
        self.param_funcs["check_segmentation"] = self.check_segm.isChecked
        self.param_funcs["ROI_list"] = self.ROI_Widget.getValue
        self.param_funcs["ROI_mask"] = self.ROI_Widget.getMask
        self.param_funcs["use_bkg"] = self.bkg_widget.CheckBox.isChecked
        self.param_funcs["bkg_model"] = self.bkg_widget.getBkg
        self.param_funcs["setup_points"] = self.setup_widget.getValue
        self.param_funcs["video_paths"] = self.open_widget.getVideoPaths
        self.param_funcs["video_fps"] = self.open_widget.getFps
        self.param_funcs["video_n_frames"] = self.open_widget.getNframes
        self.param_funcs["episodes"] = self.open_widget.getEpisodes
        self.param_funcs["video_size"] = self.open_widget.getSize
        self.param_funcs["session"] = self.getSessionName
        self.param_funcs["track_wo_identities"] = self.track_wo_id.isChecked

    def close_and_track_video(self):
        for key, item in self.param_funcs.items():
            self.GUI_out_params[key] = item()

        # signal to start tracking after closing app
        self.GUI_out_params["run_idtrackerai"] = True
        self.close()

    def getSessionName(self) -> str:
        session_name = self.session.text()
        if not session_name:
            return "no_name"
        return session_name

    def save_parameters_func(self):

        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameter file",
            str(Path.cwd() / (self.param_funcs["session"]() + ".toml")),
            filter="TOML (*.toml)",
        )

        tracking_intervals = self.param_funcs["tracking_intervals"]()
        intensity_ths = self.param_funcs["intensity_ths"]()
        area_ths = self.param_funcs["area_ths"]()
        number_of_animals = self.param_funcs["number_of_animals"]()
        resolution_reduction = self.param_funcs["resolution_reduction"]()
        check_segmentation = self.param_funcs["check_segmentation"]()
        ROI_list = self.param_funcs["ROI_list"]()
        use_bkg = self.param_funcs["use_bkg"]()
        setup_points = self.param_funcs["setup_points"]()
        video_paths = self.param_funcs["video_paths"]()
        session = self.param_funcs["session"]()
        track_wo_identities = self.param_funcs["track_wo_identities"]()

        with open(fileName, "w") as file:
            file.write(f"{session = }\n")
            file.write("video_paths" + toml_format(video_paths))
            file.write(f"{tracking_intervals = }\n")
            file.write(f"{intensity_ths = }\n")
            file.write(f"{area_ths = }\n")
            file.write(f"{number_of_animals = }\n")
            file.write("use_bkg" + toml_format(use_bkg))
            file.write(f"{resolution_reduction = }\n")
            file.write("check_segmentation" + toml_format(check_segmentation))
            file.write(
                "track_wo_identities" + toml_format(track_wo_identities)
            )
            file.write("ROI_list" + toml_format(ROI_list))
            file.write("setup_points" + toml_format(setup_points))

    def keyPressEvent(self, event: QKeyEvent):
        if hasattr(event, "isAutoRepeat") and event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key_Q:
            QCoreApplication.quit()
        if key in (Qt.Key_Enter, Qt.Key_Return):
            self.ROI_Widget.enter_key_event()
            self.setup_widget.enter_key_event()
        else:
            self.VideoPlayer.redirect_keyPressEvent(event.text().lower())

    def keyReleaseEvent(self, event: QKeyEvent):
        if hasattr(event, "isAutoRepeat"):
            if event.isAutoRepeat():
                return
        key = event.text().lower()
        self.VideoPlayer.redirect_keyReleaseEvent(key)

    def clearFocus(self):
        focused_widged = QApplication.focusWidget()
        if focused_widged:
            focused_widged.clearFocus()

    def mousePressEvent(self, event):
        self.clearFocus()
        super().mousePressEvent(event)

    @staticmethod
    def get_list_of_widgets(layout: QBoxLayout) -> list[QWidget]:
        widgets = []
        layouts = [layout]
        while layouts:
            element = layouts.pop()
            if hasattr(element.widget(), "setEnabled"):
                widgets.append(element.widget())
            else:
                layouts += [element.itemAt(i) for i in range(element.count())]
        return widgets

    def new_video_paths(self, video_paths):
        # FIXME
        self.VideoPlayer.setEnabled(False)
        self.tracking_interval.reset(self.param_funcs["video_n_frames"]())
        self.VideoPlayer.update_video_paths(
            video_paths,
            self.param_funcs["video_n_frames"](),
            self.param_funcs["video_size"](),
            self.open_widget.getFps(),
        )
        if not self.enabled:
            for widget in self.list_of_widgets:
                widget.setEnabled(True)
            self.enabled = True

        self.VideoPlayer.setEnabled(True)
        # self.bkg_widget.reset()
        self.ROI_Widget.ListChanged.emit()


def toml_format(l: list[str] | bool, width=50) -> str:
    if isinstance(l, bool):
        return " = true\n" if l else " = false\n"

    if not l:
        return " = []\n"

    if len(l) == 1:
        if len(l[0]) < width:
            return f' = ["{l[0]}"]\n'
        else:
            return f' = [\n    "{l[0]}"\n]\n'
    else:
        s = " = [\n"
        for item in l:
            s += f'    "{item}",\n'
        s += "]\n"
        return s
