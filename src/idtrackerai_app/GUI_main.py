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
    QProgressBar,
    QTextEdit,
    QSizePolicy,
    QStyle,
    QBoxLayout,
)
from confapp import conf
from PyQt6.QtCore import Qt, QCoreApplication
from matplotlib.backend_bases import KeyEvent as matplotlib_KeyEvent
from PyQt6.QtGui import QKeyEvent as PyQt_KeyEvent
import os
from idtrackerai_app.GUI_Widgets import (
    VideoPlayer,
    ROI_Widget,
    SetupPointsWidget,
    OpenBtnWidget,
    background_row,
    my_QLabeleRangeSlider,
    TrackingIntervalWidget,
)
import logging
import json

logger = logging.getLogger(__name__)


class Window(QWidget):
    def __init__(self, GUI_out_params):

        logger.debug("Initializing GUI")
        super().__init__()

        self.setWindowTitle("idTracker.ai | segmentation GUI")
        self.setGeometry(100, 60, 1000, 800)
        self.GUI_out_params = GUI_out_params
        self.param_funcs = {}

        self.open_widget = OpenBtnWidget()
        self.open_widget.button_open.clicked.connect(self.enable_all)

        ##### Resolution reduction #####
        self.resreduct = QSpinBox(
            maximum=100,
            minimum=10,
            singleStep=10,
            suffix="%",
            value=int(conf.RES_REDUCTION_DEFAULT * 100),
            enabled=False,
        )
        self.resreduct.editingFinished.connect(self.remove_any_focus)

        ##### NUMBER OF ANIMALS #####
        self.number_of_animals_widget = QSpinBox(
            maximum=100,
            minimum=1,
            value=int(conf.NUMBER_OF_ANIMALS_DEFAULT),
        )
        self.number_of_animals_widget.setKeyboardTracking(False)
        self.number_of_animals_widget.editingFinished.connect(
            self.remove_any_focus
        )

        ##### Show segmented blobs information #####
        self.Segmented_blobs_info_widget = QCheckBox("Segmented blobs info")
        self.Segmented_blobs_info_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.Segmented_blobs_info_widget.stateChanged.connect(
            lambda state: self.VideoPlayer.area_chart_widget.canvas.setVisible(
                state
            )
        )

        ##### Check segmentation #####
        self.Check_segmentation_widget = QCheckBox("Check segmentation")
        self.Check_segmentation_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        ##### Background Subtraction #####
        self.bkg_widget = background_row(self.param_funcs)

        ##### Intensity thresholds #####
        self.VideoPlayer = VideoPlayer(self.param_funcs)
        self.intensity_thresholds = my_QLabeleRangeSlider(
            "Intensity thresholds",
            conf.MIN_THRESHOLD,
            conf.MAX_THRESHOLD,
            conf.MIN_THRESHOLD_DEFAULT,
            conf.MAX_THRESHOLD_DEFAULT,
            self.VideoPlayer.new_params,
        )
        self.area_thresholds = my_QLabeleRangeSlider(
            "Area thresholds",
            conf.AREA_LOWER,
            conf.AREA_UPPER,
            conf.MIN_AREA_DEFAULT,
            conf.MAX_AREA_DEFAULT,
            self.VideoPlayer.new_params,
        )

        ##### Tracking interval ####

        self.tracking_interval = TrackingIntervalWidget()

        ##### Session #####
        self.session = QTextEdit()
        self.session.setPlaceholderText("Example: text, experiment_32A, ...")
        self.session.setFixedHeight(28)
        self.save_parameters = QPushButton("Save parameters")
        self.save_parameters.clicked.connect(self.save_parameters_func)

        self.track_wo_id = QCheckBox("Track without identities")

        ##### Add setup info #####

        self.setup_widget = SetupPointsWidget()

        main_box = QHBoxLayout()
        right = QVBoxLayout()
        left = QVBoxLayout()
        self.setLayout(main_box)
        main_box.addLayout(left)
        main_box.addLayout(right)

        self.ROI_Widget = ROI_Widget(self.param_funcs)

        video_row = QHBoxLayout()
        video_row.addWidget(self.open_widget)
        video_row.addWidget(QLabel("Resolution reduction"))
        video_row.addWidget(self.resreduct)

        left.addLayout(video_row)

        left.addLayout(self.tracking_interval.layout)
        left.addLayout(self.ROI_Widget)
        left.addLayout(self.bkg_widget)
        row_1 = QHBoxLayout()
        row_1.addWidget(QLabel("Number of animals"))
        row_1.addWidget(self.number_of_animals_widget)
        row_1.addWidget(self.Segmented_blobs_info_widget)
        row_1.addWidget(self.Check_segmentation_widget)
        left.addLayout(row_1)
        intensity_row = QHBoxLayout()
        intensity_row.addLayout(self.intensity_thresholds)
        left.addLayout(intensity_row)
        area_row = QHBoxLayout()
        area_row.addLayout(self.area_thresholds)
        left.addLayout(area_row)
        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("Session"))
        session_row.addWidget(self.session)
        session_row.addWidget(self.save_parameters)
        left.addLayout(self.setup_widget)

        left.addWidget(self.track_wo_id)
        left.addLayout(session_row)

        self.track_btn = QPushButton("Close and track video")
        self.track_btn.clicked.connect(self.close_and_track_video)
        left.addWidget(self.track_btn)

        self.build_param_funcs()

        self.ROI_Widget.add_ax_reference(self.VideoPlayer.ax)
        self.ROI_Widget.draw_and_flush = self.VideoPlayer.draw_and_flush
        self.setup_widget.add_ax_reference(self.VideoPlayer.ax)
        self.setup_widget.draw_and_flush = self.VideoPlayer.draw_and_flush
        self.bkg_widget.thread.finished.connect(self.VideoPlayer.new_params)
        self.setup_widget.list.model().rowsInserted.connect(
            self.share_updated_setup
        )
        self.setup_widget.list.model().rowsRemoved.connect(
            self.share_updated_setup
        )
        self.setup_widget.CheckBox.stateChanged.connect(
            self.share_updated_setup
        )

        self.ROI_Widget.update_mask_patches_on_VideoPlayer = (
            self.VideoPlayer.update_mask
        )

        self.VideoPlayer.click_in_plt_button_1 = self.click_in_plt_button_1

        right.addLayout(self.VideoPlayer.VideoPlayer_layout)

        self.VideoPlayer.fig.canvas.mpl_connect(
            "key_release_event", self.keyPressEvent
        )
        self.VideoPlayer.fig.canvas.setFocus()

        self.creating_ROI = False
        self.list_of_widgets = self.get_list_of_widgets(self.layout())
        for widget in self.list_of_widgets:
            widget.setEnabled(False)
        self.open_widget.setEnabled(True)

    def none_func(self):
        return None

    def build_param_funcs(self):
        self.param_funcs["open-multiple-files"] = self.none_func
        self.param_funcs["session"] = self.session.toPlainText
        self.param_funcs["tracking_interval"] = self.tracking_interval.value
        self.param_funcs["intensity_ths"] = self.intensity_thresholds.value
        self.param_funcs["area_ths"] = self.area_thresholds.value
        self.param_funcs[
            "number_of_animals"
        ] = self.number_of_animals_widget.value
        self.param_funcs["resolution_reduction"] = self.resreduct.value
        self.param_funcs[
            "check_segmentation"
        ] = self.Check_segmentation_widget.isChecked
        self.param_funcs["ROI_list"] = self.ROI_Widget.str_list
        self.param_funcs["ROI_mask"] = self.ROI_Widget.get_mask
        self.param_funcs["no_ids"] = self.track_wo_id.isChecked
        self.param_funcs["bkg_check"] = self.bkg_widget.CheckBox.isChecked
        self.param_funcs["bkg_model"] = self.bkg_widget.get_bkg
        self.param_funcs["setup_points"] = self.none_func
        self.param_funcs["video_paths"] = self.open_widget.video_paths
        self.param_funcs["episodes"] = self.open_widget.episodes
        self.param_funcs["video_height"] = self.open_widget.video_height
        self.param_funcs["video_width"] = self.open_widget.video_width
        self.param_funcs["ROI_patches"] = self.ROI_Widget.get_patches
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
        session_name = self.session.toPlainText()
        if not session_name:
            return "no_name"
        return session_name

    def save_parameters_func(self):
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameter file",
            os.path.join(os.getcwd(), self.param_funcs["session"]() + ".json"),
            filter="JSON (*.json)",
        )

        if fileName[-5:] != ".json":
            fileName += ".json"

        keys_to_ignore = (
            "ROI_mask",
            "bkg_model",
            "episodes",
            "video_height",
            "video_width",
            "ROI_patches",
        )

        dict_to_print = {
            key: value() for key, value in self.param_funcs.items()
        }

        for key in keys_to_ignore:
            if key in dict_to_print:
                dict_to_print.pop(key)

        with open(fileName, "w") as file:
            json.dump(dict_to_print, file, indent=4)

    def remove_any_focus(self):
        focused_widged = QApplication.focusWidget()
        if focused_widged:
            focused_widged.clearFocus()

    def keyPressEvent(self, event):
        if isinstance(event, matplotlib_KeyEvent):
            key = event.key
        elif isinstance(event, PyQt_KeyEvent):
            key = event.text()
        else:
            print("Not known key event")

        print(key, "pressed")
        if key == "q":
            QCoreApplication.quit()
        if key == "enter":
            self.ROI_Widget.enter_key_event()
            self.setup_widget.enter_key_event()

        else:
            self.VideoPlayer.redirect_keyPressEvent(key)

    def click_in_plt_button_1(self, event):
        if self.ROI_Widget.add.isChecked():
            self.ROI_Widget.click_event(event)
        if self.setup_widget.add.isChecked():
            self.setup_widget.click_event(event)

    def mousePressEvent(self, event):
        self.remove_any_focus()
        QWidget.mousePressEvent(self, event)

    @staticmethod
    def get_list_of_widgets(layout):
        widgets = []
        layouts = [layout]
        while layouts:
            element = layouts.pop()
            if isinstance(element, QBoxLayout):
                for subelement in (
                    element.itemAt(i) for i in range(element.count())
                ):
                    layouts.append(subelement)
            else:
                widgets.append(element.widget())
        return widgets

    def enable_all(self):
        video_paths = self.param_funcs["video_paths"]()
        if video_paths:
            for widget in self.list_of_widgets:
                widget.setEnabled(True)
            self.VideoPlayer.update_video(video_paths[0])
            self.tracking_interval.update_ranges(
                0, self.VideoPlayer.video_holder.n_frames
            )

    def share_updated_setup(self):
        legend_needed = (
            self.setup_widget.CheckBox.isChecked()
            and self.setup_widget.list.count()
        )
        if legend_needed:
            self.VideoPlayer.ax.legend()
        else:
            self.VideoPlayer.ax.legend([]).set_visible(False)
        self.VideoPlayer.draw_and_flush()
