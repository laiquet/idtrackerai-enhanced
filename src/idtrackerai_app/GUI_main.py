from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QSpinBox,
    QLineEdit,
)
from matplotlib.pyplot import rcParams
from confapp import conf
from PyQt6.QtCore import Qt, QCoreApplication
from matplotlib.backend_bases import KeyEvent as matplotlib_KeyEvent
from PyQt6.QtGui import QKeyEvent as PyQt_KeyEvent
from pathlib import Path
from idtrackerai_app.GUI_Widgets import (
    VideoPlayerWidget,
    ROIWidget,
    SetupPointsWidget,
    OpenVideoWidget,
    BkgWidget,
    TrackingIntervalsWidget,
    BlobInfoWidget,
)
from idtrackerai_app.widgets_utils import LabelRangeSlider
import logging
import json


class Window(QWidget):
    def __init__(self, GUI_out_params):

        logging.debug("Initializing GUI")
        super().__init__()

        # Clean all the default keyboard shortcuts of matplotlib
        for action, keybindings in rcParams.items():
            if action.startswith("keymap."):
                keybindings.clear()
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = "Arial"

        self.setWindowTitle("idTracker.ai | segmentation GUI")
        self.setGeometry(100, 60, 1000, 800)
        self.GUI_out_params = GUI_out_params
        self.param_funcs = {}

        self.open_widget = OpenVideoWidget(self)
        self.VideoPlayer = VideoPlayerWidget(self, self.param_funcs)
        self.BlobInfo = BlobInfoWidget()
        self.bkg_widget = BkgWidget(self, self.param_funcs)
        self.tracking_interval = TrackingIntervalsWidget(parent=self)
        self.open_widget.path_clicked.connect(self.VideoPlayer.setCurrentFrame)
        self.open_widget.new_video_paths.connect(self.new_video_paths)
        self.open_widget.new_video_paths.connect(self.bkg_widget.reset)
        self.open_widget.new_video_paths.connect(self.VideoPlayer.update_mask)
        self.open_widget.video_paths_reordered.connect(self.bkg_widget.reset)
        self.open_widget.video_paths_reordered.connect(
            self.VideoPlayer.reorder_video_paths
        )

        self.resreduct = QSpinBox(
            maximum=100,
            minimum=10,
            singleStep=10,
            suffix="%",
            value=int(conf.RES_REDUCTION_DEFAULT * 100),
        )
        self.resreduct.editingFinished.connect(self.resreduct.clearFocus)
        self.resreduct.valueChanged.connect(self.VideoPlayer.new_params)

        self.check_segm = QCheckBox("Check segmentation")
        self.check_segm.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.number_of_animals = QSpinBox(
            maximum=100,
            minimum=1,
        )
        self.number_of_animals.editingFinished.connect(
            self.number_of_animals.clearFocus
        )
        # TODO there won't be .set method
        self.number_of_animals.valueChanged.connect(
            lambda n: self.BlobInfo.set(n_animals=n)
        )
        self.tracking_interval.has_changed.connect(
            lambda trk_int: self.BlobInfo.set(tracking_intervals=trk_int)
        )
        self.intensity_thresholds = LabelRangeSlider(
            min=conf.MIN_THRESHOLD, max=conf.MAX_THRESHOLD
        )
        self.intensity_thresholds.has_changed.connect(
            self.VideoPlayer.new_params
        )

        self.area_thresholds = LabelRangeSlider(
            min=conf.AREA_LOWER, max=conf.AREA_UPPER
        )
        self.area_thresholds.has_changed.connect(self.VideoPlayer.new_params)

        self.tracking_interval.has_changed.connect(self.bkg_widget.reset)

        self.session = QLineEdit()
        self.session.setPlaceholderText("Example: text, experiment_32A, ...")
        self.session.setFixedHeight(28)

        self.session.editingFinished.connect(self.session.clearFocus)
        self.save_parameters = QPushButton("Save parameters")
        self.save_parameters.clicked.connect(self.save_parameters_func)

        self.track_wo_id = QCheckBox("Track without identities")
        self.setup_widget = SetupPointsWidget()
        self.ROI_Widget = ROIWidget(self.param_funcs)

        QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.layout().addLayout(left, 40)
        self.layout().addLayout(right, 60)
        left.addLayout(self.open_widget)
        res_reduct_row = QHBoxLayout()
        res_reduct_row.addWidget(QLabel("Resolution reduction"))
        res_reduct_row.addWidget(self.resreduct)
        left.addLayout(res_reduct_row)
        left.addLayout(self.tracking_interval)
        left.addLayout(self.ROI_Widget)
        left.addLayout(self.bkg_widget)
        row_1 = QHBoxLayout()
        row_1.addWidget(QLabel("Number of animals"))
        row_1.addWidget(self.number_of_animals)
        row_1.addWidget(self.check_segm)
        left.addLayout(row_1)
        intensity_row = QHBoxLayout()
        intensity_row.addWidget(QLabel("Intensity thresholds"))
        intensity_row.addWidget(self.intensity_thresholds)
        left.addLayout(intensity_row)
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Area thresholds"))
        area_row.addWidget(self.area_thresholds)
        left.addLayout(area_row)
        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("Session"))
        session_row.addWidget(self.session)
        session_row.addWidget(self.save_parameters)
        left.addLayout(self.setup_widget)
        left.addWidget(self.track_wo_id)
        left.addLayout(session_row)

        self.track_btn = QPushButton("Close window and track video")
        # self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.track_btn.clicked.connect(self.close_and_track_video)
        left.addWidget(self.track_btn)

        self.build_param_funcs()

        self.ROI_Widget.add_ax_reference(self.VideoPlayer.canvas.ax)
        self.ROI_Widget.draw_and_flush.connect(
            self.VideoPlayer.canvas.draw_and_flush
        )
        self.ROI_Widget.ListChanged.connect(self.VideoPlayer.update_mask)
        self.ROI_Widget.ListChanged.connect(self.bkg_widget.ROI_has_updated)

        self.setup_widget.add_ax_reference(self.VideoPlayer.canvas.ax)
        self.bkg_widget.new_bkg_data.connect(self.VideoPlayer.new_params)
        self.setup_widget.ListChanged.connect(
            self.VideoPlayer.canvas.draw_and_flush
        )
        self.setup_widget.draw_and_flush.connect(
            self.VideoPlayer.canvas.draw_and_flush
        )
        self.VideoPlayer.new_areas.connect(self.BlobInfo.setAreas)

        self.VideoPlayer.canvas.click_on_plot.connect(
            self.ROI_Widget.click_event
        )
        self.VideoPlayer.canvas.click_on_plot.connect(
            self.setup_widget.click_event
        )

        right.addLayout(self.BlobInfo, 30)
        right.addLayout(self.VideoPlayer, 70)

        self.VideoPlayer.canvas.mpl_connect(
            "key_release_event", self.keyPressEvent
        )

        self.creating_ROI = False
        self.list_of_widgets = self.get_list_of_widgets(self.layout())
        for widget in self.list_of_widgets:
            widget.setEnabled(False)
        self.enabled = False
        self.open_widget.setEnabled(True)

        self.load_parameters(self.GUI_out_params)

        self.setTabOrder(self.resreduct, self.VideoPlayer.canvas)
        self.setTabOrder(self.VideoPlayer.canvas, self.resreduct)

    def load_parameters(self, load_dict: dict):

        resolution_reduction = load_dict.get("resolution_reduction", 1)
        self.resreduct.setValue(int(resolution_reduction * 100))

        self.tracking_interval.setValue(
            load_dict.get("tracking_intervals", None)
        )

        self.setup_widget.setValue(load_dict.get("setup_points", None))

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

        self.number_of_animals.setValue(
            load_dict.get("number_of_animals", conf.NUMBER_OF_ANIMALS_DEFAULT)
        )

        self.track_wo_id.setChecked(
            load_dict.get("track_wo_identification", False)
        )

        self.check_segm.setChecked(load_dict.get("check_segmentation", False))
        self.session.setText(load_dict.get("session", ""))

        self.open_widget.open_video_paths(
            video_paths=load_dict.get("video_paths", None)
        )
        if load_dict.get("use_bkg", False):
            self.bkg_widget.CheckBox.click()
        # self.VideoPlayer.new_params()

    def build_param_funcs(self):
        self.param_funcs["tracking_intervals"] = self.tracking_interval.value
        self.param_funcs["intensity_ths"] = self.intensity_thresholds.value
        self.param_funcs["area_ths"] = self.area_thresholds.value
        self.param_funcs["number_of_animals"] = self.number_of_animals.value
        self.param_funcs["resolution_reduction"] = (
            lambda: self.resreduct.value() / 100
        )
        self.param_funcs["check_segmentation"] = self.check_segm.isChecked
        self.param_funcs["ROI_list"] = self.ROI_Widget.str_list
        self.param_funcs["ROI_mask"] = self.ROI_Widget.getMask
        self.param_funcs["ROI_patches"] = self.ROI_Widget.getPatches
        self.param_funcs["no_ids"] = self.track_wo_id.isChecked
        self.param_funcs["use_bkg"] = self.bkg_widget.CheckBox.isChecked
        self.param_funcs["bkg_model"] = self.bkg_widget.get_bkg
        self.param_funcs["setup_points"] = self.setup_widget.readList
        self.param_funcs["video_paths"] = self.open_widget.getVideoPaths
        self.param_funcs["video_fps"] = self.open_widget.getFps
        self.param_funcs["video_n_frames"] = self.open_widget.getNframes
        self.param_funcs["episodes"] = self.open_widget.getEpisodes
        self.param_funcs["video_size"] = self.open_widget.getSize
        self.param_funcs["session"] = self.get_session_name
        self.param_funcs[
            "track_wo_identification"
        ] = self.track_wo_id.isChecked

    def close_and_track_video(self):
        for key, item in self.param_funcs.items():
            self.GUI_out_params[key] = item()

        # signal to start tracking after closing app
        self.GUI_out_params["run_idtrackerai"] = True
        self.close()

    def get_session_name(self):
        session_name = self.session.text()
        if not session_name:
            return "no_name"
        return session_name

    def save_parameters_func(self):
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameter file",
            str(Path.cwd() / (self.param_funcs["session"]() + ".json")),
            filter="JSON (*.json)",
        )

        if fileName[-5:] != ".json":
            fileName += ".json"

        keys_to_ignore = (
            "ROI_mask",
            "bkg_model",
            "episodes",
            "video_size",
            "ROI_patches",
            "video_fps",
            "video_n_frames",
        )

        dict_to_print = {
            key: value() for key, value in self.param_funcs.items()
        }

        for key in keys_to_ignore:
            if key in dict_to_print:
                dict_to_print.pop(key)

        with open(fileName, "w") as file:
            json.dump(dict_to_print, file, indent=4)

    def keyPressEvent(self, event):
        if isinstance(event, matplotlib_KeyEvent):
            key = event.key
        elif isinstance(event, PyQt_KeyEvent):
            key = event.text()
        else:
            logging.info("Not known key event")

        if key == "q":
            QCoreApplication.quit()
        if key == "enter":
            self.ROI_Widget.enter_key_event()
            self.setup_widget.enter_key_event()
        elif self.enabled:
            self.VideoPlayer.redirect_keyPressEvent(key)

    def mousePressEvent(self, event):
        focused_widged = QApplication.focusWidget()
        if focused_widged:
            focused_widged.clearFocus()
        super().mousePressEvent(event)

    @staticmethod
    def get_list_of_widgets(layout):
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
        if not self.enabled:
            for widget in self.list_of_widgets:
                widget.setEnabled(True)
            self.enabled = True
        self.tracking_interval.reset(self.param_funcs["video_n_frames"]())
        self.VideoPlayer.update_video_paths(video_paths)


# TODO checkbosex have the box too dark, barely visible
