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
from idtrackerai.animals_detection.segmentation import _process_frame

# from matplotlib.patches import Polygon
from superqt import QLabeledRangeSlider, QLabeledDoubleRangeSlider
from .GUI_video_player import VideoPlayer
from .ROI_widget import ROI_Widget
import logging
import numpy as np
import json
import cv2
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


# Plots a Polygon to pyplot `ax`
def shapely_poly_to_mpl_patch(poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
    )
    return PathPatch(path, **kwargs)


def create_mask(text, height, width):
    mask = np.zeros((width, height), np.uint8)
    main_poly = Polygon([[0, 0], [0, width], [height, width], [height, 0]])
    for line in text.splitlines():
        if line[0] == "P":
            main_poly = main_poly.difference(Polygon(json.loads(line[2:])))
        elif line[0] == "N":
            main_poly = main_poly.union(Polygon(json.loads(line[2:])))
        else:
            raise TypeError

    if isinstance(main_poly, Polygon):
        patches = [shapely_poly_to_mpl_patch(main_poly, color="r", alpha=0.5)]
    else:
        # if it is not a Polygon, it is a collection of Polygons
        patches = [
            shapely_poly_to_mpl_patch(polygon, color="r", alpha=0.5)
            for polygon in main_poly.geoms
        ]

    return mask, patches


class Window(QWidget, VideoPlayer):
    def __init__(self):

        logger.debug("Initializing GUI")
        # super().__init__()
        QWidget.__init__(self)
        VideoPlayer.__init__(self)
        # ROI_Widget.__init__(self)

        # ROI_Widget.ax = self.ax

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
            value=conf.RES_REDUCTION_DEFAULT,
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
            lambda state: self.area_chart_widget.canvas.setVisible(state)
        )

        ##### Check segmentation #####
        self.Check_segmentation_widget = QCheckBox("Check segmentation")
        self.Check_segmentation_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        ##### Background Subtraction #####
        self.Subtract_bkg = QCheckBox("Subtract background")
        self.Subtract_bkg.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.Subtract_bkg.stateChanged.connect(
            lambda state: self.bkg_pbar.setEnabled(state)
        )

        ##### Background progress bar
        self.bkg_pbar = QProgressBar()

        ##### Intensity thresholds #####
        self.intensity_thresholds = QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal
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
        self.tracking_interval_label = QLabel("Tracking interval")
        self.tracking_interval = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.tracking_interval.setEnabled(False)
        self.tracking_interval.setFixedHeight(40)

        self.multiple_range = QCheckBox("Multiple", enabled=False)

        def multiple_range_change_state(state):
            self.tracking_interval_label.setText(
                "Tracking interval" + bool(state) * "s"
            )
            self.tracking_interval.setVisible(not state)
            self.add_interval.setVisible(state)
            self.multiple_ranges.setVisible(state)

        self.multiple_range.stateChanged.connect(multiple_range_change_state)
        self.multiple_range.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.multiple_ranges = QTextEdit(visible=False)
        self.multiple_ranges.setPlaceholderText(
            "Example: [0,1000],[1300,2400],..."
        )
        self.multiple_ranges.setFixedHeight(28)
        self.add_interval = QPushButton("Add interval", visible=False)

        ##### Session #####
        self.session = QTextEdit("test")
        self.session.setFixedHeight(28)
        self.save_parameters = QPushButton("Save parameters")

        self.track_wo_id = QCheckBox("Track without identities")

        ##### Add setup info #####

        self.setup_check = QCheckBox("Add setup info")
        self.setup_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        def setup_check_changed(state):
            self.setup_list.setVisible(state)
            self.add_setup.setEnabled(state)
            self.remove_setup.setEnabled(
                state and len(self.setup_list.selectedItems())
            )

        self.setup_check.stateChanged.connect(setup_check_changed)

        def add_setup_func():
            self.setup_list.addItem(f"item number {self.setup_list.count()}")

        self.add_setup = QPushButton("Add setup", enabled=False)
        self.add_setup.clicked.connect(add_setup_func)
        self.remove_setup = QPushButton("Remove selected", enabled=False)

        def remove_setup_func():
            for item in self.setup_list.selectedItems():
                self.setup_list.takeItem(self.setup_list.row(item))
            if not len(self.setup_list.selectedItems()):
                self.remove_setup.setEnabled(False)

        self.remove_setup.clicked.connect(remove_setup_func)
        self.setup_list = QListWidget(visible=False)
        self.setup_list.addItem("control_item")
        self.setup_list.setFixedHeight(
            self.setup_list.sizeHintForRow(0) * 5
            + 2 * self.setup_list.frameWidth(),
        )
        self.setup_list.clear()

        self.setup_list.itemClicked.connect(
            lambda: self.remove_setup.setEnabled(
                self.setup_check.isChecked()
                and len(self.setup_list.selectedItems())
            )
        )

        setup_VBox = QVBoxLayout()
        setup_Controls_HBox = QHBoxLayout()
        setup_Controls_HBox.addWidget(self.setup_check)
        setup_Controls_HBox.addWidget(self.add_setup)
        setup_Controls_HBox.addWidget(self.remove_setup)

        setup_VBox.addLayout(setup_Controls_HBox)
        setup_VBox.addWidget(self.setup_list)

        # self.ROI_Widget = ROI_Widget()

        main_box = QHBoxLayout()
        right = QVBoxLayout()
        left = QVBoxLayout()
        self.setLayout(main_box)
        main_box.addLayout(left)
        main_box.addLayout(right)

        self.ROI_Widget = ROI_Widget()
        (self.ROI_Widget.building_ROI,) = self.ax.plot([], [], ".-")

        video_row = QHBoxLayout()
        self.label_video = QLabel("Video:")
        video_row.addWidget(self.label_video)
        video_row.addWidget(self.button_open)
        video_row.addWidget(QLabel("Resolution reduction"))
        video_row.addWidget(self.resreduct)
        left.addLayout(video_row)
        left.addLayout(self.ROI_Widget.ROI_Layout)
        bkg_row = QHBoxLayout()
        bkg_row.addWidget(self.Subtract_bkg)
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
        left.addLayout(setup_VBox)

        left.addWidget(self.track_wo_id)

        tracking_interval_row = QHBoxLayout()
        tracking_interval_row.addWidget(self.tracking_interval_label)
        tracking_interval_row.addWidget(self.tracking_interval)
        tracking_interval_row.addWidget(self.multiple_ranges)
        tracking_interval_row.addWidget(self.add_interval)
        tracking_interval_row.addWidget(self.multiple_range)
        left.addLayout(tracking_interval_row)
        left.addLayout(session_row)

        self.layouts = [
            video_row,
            bkg_row,
            row_1,
            intensity_row,
            area_row,
            tracking_interval_row,
        ]
        for layout in self.layouts:
            for widget in (
                layout.itemAt(i).widget() for i in range(layout.count())
            ):
                widget.setEnabled(False)
        self.label_video.setEnabled(True)
        self.button_open.setEnabled(True)

        right.addLayout(self.VideoPlayer_layout)

        self.fig.canvas.mpl_connect("key_release_event", self.keyPressEvent)

        self.fig.canvas.setFocus()
        # self.button_open.setEnabled(True)
        # super().setEnabled(False)

        self.creating_ROI = False
        self.button_open_clicked(opened="/home/jordi/fish_video.MP4")

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
        if key == "q":
            QCoreApplication.quit()
        if key == "enter":
            if self.ROI_Widget.ROI_mode_isactive:
                ROIS_text = self.ROI_Widget.end_ROI_mode()
                self.mask, polygons = create_mask(
                    ROIS_text,
                    self.video_holder.width,
                    self.video_holder.height,
                )
                self.update_mask(polygons)
                self.draw_and_flush()

        else:
            print(key)
            self.redirect_keyPressEvent(key)

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
            self.ROI_Widget.setROIEnabled(True)
            self.update_video(fileName)
            self.tracking_interval.setRange(0, self.video_holder.n_frames)
            self.tracking_interval.setValue(
                (0, int(self.video_holder.n_frames))
            )

    def click_in_plt_button_1(self, event):
        if self.ROI_Widget.ROI_mode_isactive:
            self.ROI_Widget.click_event(event)
            self.draw_and_flush()
            # xy = self.building_ROI.get_xydata()
            # self.building_ROI.set_data(np.vstack([xy, (event.x, event.y)]).T)

    def mousePressEvent(self, event):
        self.remove_any_focus()
        QWidget.mousePressEvent(self, event)

    def process_frame_evt(self, frame):
        """
        Function called before an image is shown in the player.
        It does the pre-visualization segmentation and ROIs selection.
        """
        # Save original shape to rescale if resolution reduction is applied
        original_size = frame.shape[1], frame.shape[0]  # (width, height)
        self._frame_width = original_size[0]
        self._frame_height = original_size[1]
        # TODO: check if bkgmodel needs to be updated because of new ROI
        self._mask_img = self.create_mask(
            self._frame_height, self._frame_width
        )
        animal_detection_parameters = {
            "number_of_animals": int(self.number_of_animals_widget.value),
            "min_threshold": self.intensity_thresholds.value()[0],
            "max_threshold": self.intensity_thresholds.value()[1],
            "min_area": self.area_thresholds.value()[0],
            "max_area": self.area_thresholds.value()[1],
            "tracking_interval": None,
            "apply_ROI": False,  # self.ROI_check.isChecked(),
            "rois": None,  # self._roi.value,
            "mask": self._mask_img,
            "subtract_bkg": False,  # self.Subtract_bkg.isChecked(),
            "bkg_model": None,  # self._background_img,
            "resolution_reduction": self.resreduct.value(),
            "sigma_gaussian_blurring": conf.SIGMA_GAUSSIAN_BLURRING,
        }

        (boxes, _, _, areas, _, contours, _,) = _process_frame(
            frame,
            animal_detection_parameters,
            -1,  # Get frame_number from the Widget
            save_pixels="NONE",
            save_segmentation_image="NONE",
        )

        # Save detected areas to plot the bar plot of the blobs size in pixels
        self._detected_areas = areas
        # Update graph with areas of seglemted blobs
        if conf.PYFORMS_MODE == "GUI" and self._toggle_blobs_area_info.value:
            self._graph.draw()
        # Draw detected blobs in frame
        if animal_detection_parameters["resolution_reduction"] != 1:
            frame = cv2.resize(
                frame,
                None,
                fx=animal_detection_parameters["resolution_reduction"],
                fy=animal_detection_parameters["resolution_reduction"],
                interpolation=cv2.INTER_AREA,
            )
        cv2.drawContours(frame, contours, -1, color=(0, 0, 255), thickness=-1)
        # Resize to original size (ROI and setup points are in original size)
        frame = cv2.resize(frame, original_size, interpolation=cv2.INTER_AREA)
        # Draw ROIs in frame
        self.draw_rois(frame)
        # Draw setup points in frame
        self.draw_points_list(frame)
        return frame


# print("la vida segueix")
