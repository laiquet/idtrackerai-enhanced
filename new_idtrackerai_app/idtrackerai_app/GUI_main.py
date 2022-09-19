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
    QListWidget,
    QProgressBar,
    QTextEdit,
    QSizePolicy,
    QStyle,
)
from confapp import conf
from PyQt6.QtCore import Qt, QCoreApplication
from matplotlib.backend_bases import KeyEvent as matplotlib_KeyEvent
from PyQt6.QtGui import QKeyEvent as PyQt_KeyEvent
from PyQt6.QtGui import QFont
from idtrackerai.animals_detection.segmentation import _process_frame

# from matplotlib.patches import Polygon
from superqt import QLabeledRangeSlider, QLabeledDoubleRangeSlider
from .GUI_video_player import VideoPlayer
from .ROI_widget import ROI_Widget
from .setup_points_widget import SetupPointsWidget
import logging
import numpy as np
import json
import cv2
from shapely.geometry import Polygon
from .tracking_interval_widget import TrackingIntervalWidget

logger = logging.getLogger(__name__)
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


def points_in_ellipse(ox, oy, a, b, angle):
    t = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)
    rot_x = np.cos(angle) * x - np.sin(angle) * y + ox
    rot_y = np.sin(angle) * x + np.cos(angle) * y + oy
    return np.asarray([rot_x, rot_y]).T


class Window(QWidget):
    def __init__(self):

        logger.debug("Initializing GUI")
        # super().__init__()
        QWidget.__init__(self)

        # VideoPlayer.__init__(self)
        # print(self.setFont(QFont("Impact")))

        self.setWindowTitle("idTracker.ai | segmentation GUI")
        self.setGeometry(100, 60, 1000, 800)

        ##### Open File #####
        self.button_open = QPushButton("Open")
        self.button_open.clicked.connect(self.button_open_clicked)
        self.button_open.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button_open.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )

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
        self.number_of_animals_widget.valueChanged.connect(
            self.number_of_animals_changed
        )
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
        self.subtract_bkg = QCheckBox("Subtract background")
        self.subtract_bkg.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.subtract_bkg.stateChanged.connect(
            lambda state: self.bkg_pbar.setEnabled(state)
        )

        ##### Background progress bar
        self.bkg_pbar = QProgressBar()

        ##### Intensity thresholds #####
        self.intensity_thresholds = QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal, traking=False
        )
        self.intensity_thresholds.setRange(
            conf.MIN_THRESHOLD, conf.MAX_THRESHOLD
        )
        self.intensity_thresholds.setValue(
            [conf.MIN_THRESHOLD_DEFAULT, conf.MAX_THRESHOLD_DEFAULT]
        )
        self.intensity_thresholds.setFixedHeight(40)

        ##### Area thresholds #####
        self.area_thresholds = QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal
        )
        self.area_thresholds.setEnabled(False)
        self.area_thresholds.setRange(conf.AREA_LOWER, conf.AREA_UPPER)
        self.area_thresholds.setValue(
            (conf.MIN_AREA_DEFAULT, conf.MAX_AREA_DEFAULT)
        )
        self.area_thresholds.setFixedHeight(40)

        ##### Tracking interval ####

        self.tracking_interval = TrackingIntervalWidget()

        ##### Session #####
        self.session = QTextEdit()
        self.session.setPlaceholderText("Example: text, experiment_32A, ...")
        self.session.setFixedHeight(28)
        self.save_parameters = QPushButton("Save parameters")

        self.track_wo_id = QCheckBox("Track without identities")

        ##### Add setup info #####

        self.setup_widget = SetupPointsWidget()

        main_box = QHBoxLayout()
        right = QVBoxLayout()
        left = QVBoxLayout()
        self.setLayout(main_box)
        main_box.addLayout(left)
        main_box.addLayout(right)

        self.ROI_Widget = ROI_Widget()

        # self.ROI_Widget.share_updated_ROI = self.share_updated_ROI

        self.ROI_Widget.list.model().rowsInserted.connect(
            self.share_updated_ROI
        )
        self.ROI_Widget.list.model().rowsRemoved.connect(
            self.share_updated_ROI
        )
        self.ROI_Widget.CheckBox.stateChanged.connect(self.share_updated_ROI)

        video_row = QHBoxLayout()
        self.label_video = QLabel("Video:")
        video_row.addWidget(self.label_video)
        video_row.addWidget(self.button_open)
        video_row.addWidget(QLabel("Resolution reduction"))
        video_row.addWidget(self.resreduct)
        left.addLayout(video_row)
        left.addLayout(self.ROI_Widget.Main_Layout)
        bkg_row = QHBoxLayout()
        bkg_row.addWidget(self.subtract_bkg)
        bkg_row.addWidget(self.bkg_pbar)
        left.addLayout(bkg_row)
        row_1 = QHBoxLayout()
        row_1.addWidget(QLabel("Number of animals"))
        row_1.addWidget(self.number_of_animals_widget)
        row_1.addWidget(self.Segmented_blobs_info_widget)
        row_1.addWidget(self.Check_segmentation_widget)
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
        left.addLayout(self.setup_widget.Main_Layout)

        left.addWidget(self.track_wo_id)
        left.addLayout(self.tracking_interval.layout)
        left.addLayout(session_row)

        self.layouts = [
            video_row,
            bkg_row,
            row_1,
            intensity_row,
            area_row,
            self.tracking_interval.layout,
        ]
        for layout in self.layouts:
            for widget in (
                layout.itemAt(i).widget() for i in range(layout.count())
            ):
                widget.setEnabled(False)
        self.label_video.setEnabled(True)
        self.button_open.setEnabled(True)

        self.param_funcs = self.build_param_funcs()
        self.VideoPlayer = VideoPlayer(self.param_funcs)
        self.ROI_Widget.add_ax_reference(self.VideoPlayer.ax)
        self.setup_widget.add_ax_reference(self.VideoPlayer.ax)

        self.setup_widget.list.model().rowsInserted.connect(
            self.share_updated_setup
        )
        self.setup_widget.list.model().rowsRemoved.connect(
            self.share_updated_setup
        )
        self.setup_widget.CheckBox.stateChanged.connect(
            self.share_updated_setup
        )

        self.VideoPlayer.click_in_plt_button_1 = self.click_in_plt_button_1
        self.intensity_thresholds.setTracking(False)
        self.intensity_thresholds.valueChanged.connect(
            self.VideoPlayer.new_params
        )

        right.addLayout(self.VideoPlayer.VideoPlayer_layout)

        self.VideoPlayer.fig.canvas.mpl_connect(
            "key_release_event", self.keyPressEvent
        )
        self.VideoPlayer.fig.canvas.setFocus()

        self.creating_ROI = False
        self.button_open_clicked(opened="/home/jordi/fish_video.MP4")

    def none_func(self):
        return None

    def build_param_funcs(self):
        param_funcs_dict = {}
        param_funcs_dict["open-multiple-files"] = self.none_func
        param_funcs_dict["session"] = self.session.toPlainText
        param_funcs_dict["video"] = self.button_open.text
        param_funcs_dict["range"] = self.tracking_interval.value
        param_funcs_dict["intensity"] = self.intensity_thresholds.value
        param_funcs_dict["area"] = self.area_thresholds.value
        param_funcs_dict[
            "number_of_animals"
        ] = self.number_of_animals_widget.value
        param_funcs_dict["resreduct"] = self.resreduct.value
        param_funcs_dict["chcksegm"] = self.Check_segmentation_widget.isChecked
        param_funcs_dict["ROI"] = self.ROI_Widget.str_list
        param_funcs_dict["no_ids"] = self.track_wo_id.isChecked
        param_funcs_dict["bgsub"] = self.subtract_bkg.isChecked
        param_funcs_dict["setup_info"] = self.none_func
        param_funcs_dict["mask"] = self.get_mask
        return param_funcs_dict

    def print_param_dict(self, path="data.json"):

        printing_dict = {
            key: value() for key, value in self.param_funcs.items()
        }

        with open("data.json", "w") as fp:
            json.dump(printing_dict, fp, indent=4)

    def remove_any_focus(self):
        focused_widged = QApplication.focusWidget()
        if focused_widged:
            focused_widged.clearFocus()

    def number_of_animals_changed(self):
        print(
            "number of animals changed to",
            self.number_of_animals_widget.value(),
        )

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

    # def click_in_plt_button_1(self, event):
    #     print(self)
    #     print(self.ROI_Widget.ROI_mode_isactive)
    #     if self.ROI_mode_isactive:
    #         xy = self.building_ROI.get_xydata()
    #         self.building_ROI.set_xydata(np.vstack([xy, (event.x, event.y)]))
    #     # print(f'recieved click {event.x = } {event.y = }')

    def button_open_clicked(self, opened=None):
        if opened:
            fileName = opened
        else:
            fileName, _ = QFileDialog.getOpenFileName()
        if fileName:
            self.button_open.setText(fileName)
            for layout in self.layouts:
                for widget in (
                    layout.itemAt(i) for i in range(layout.count())
                ):
                    widget.widget().setEnabled(True)
            self.ROI_Widget.set_enabled(True)
            self.setup_widget.set_enabled(True)

            self.print_param_dict()
            self.VideoPlayer.update_video(fileName)
            self.tracking_interval.update_ranges(
                0, self.VideoPlayer.video_holder.n_frames
            )

    def click_in_plt_button_1(self, event):
        if self.ROI_Widget.add.isChecked():
            self.ROI_Widget.click_event(event)
            self.VideoPlayer.draw_and_flush()
        if self.setup_widget.add.isChecked():
            self.setup_widget.click_event(event)
            self.VideoPlayer.draw_and_flush()

    def mousePressEvent(self, event):
        self.remove_any_focus()
        QWidget.mousePressEvent(self, event)

    def share_updated_ROI(self):
        """This method is called from self.ROI_Widget when its ROI_list items has changed and when the entire ROI has been enabled/disabled"""

        (width, height) = self.VideoPlayer.video_holder.size
        list_of_ROIs = self.ROI_Widget.str_list()

        if list_of_ROIs is None:
            patches = []
            self.ROI_mask = np.ones((height, width), np.uint8)
        else:

            self.ROI_mask = np.zeros((height, width), np.uint8)
            main_poly = Polygon(
                [[0, 0], [0, height], [width, height], [width, 0]]
            )
            for line in list_of_ROIs.splitlines():

                if line[2:9] == "Polygon":
                    vertices = np.asarray(json.loads(line[10:])).astype(
                        np.int32
                    )
                elif line[2:9] == "Ellipse":
                    vertices = points_in_ellipse(
                        *json.loads(line[10:])
                    ).astype(np.int32)
                else:
                    raise TypeError

                polygon = Polygon(vertices)

                if line[0] == "+":
                    main_poly = main_poly.difference(polygon)
                    cv2.fillPoly(self.ROI_mask, [vertices][::-1], color=1)
                elif line[0] == "-":
                    main_poly = main_poly.union(polygon)
                    cv2.fillPoly(self.ROI_mask, [vertices][::-1], color=0)
                else:
                    raise TypeError

            if isinstance(main_poly, Polygon):
                patches = [
                    shapely_poly_to_mpl_patch(main_poly, color="r", alpha=0.2)
                ]
            else:
                # if it is not a Polygon, it is a collection of Polygons
                patches = [
                    shapely_poly_to_mpl_patch(polygon, color="r", alpha=0.2)
                    for polygon in main_poly.geoms
                ]

        self.VideoPlayer.update_mask(patches)

    def share_updated_setup(self):
        self.VideoPlayer.ax.legend().set_visible(
            self.setup_widget.CheckBox.isChecked()
        )
        self.VideoPlayer.draw_and_flush()

    def get_mask(self):
        if self.ROI_Widget.CheckBox.isChecked():
            if self.ROI_Widget.list.count():
                return self.ROI_mask
            else:
                return 0
        else:
            return 1


# Plots a Polygon to pyplot `ax`
def shapely_poly_to_mpl_patch(poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
    )
    return PathPatch(path, **kwargs)


# print("la vida segueix")
