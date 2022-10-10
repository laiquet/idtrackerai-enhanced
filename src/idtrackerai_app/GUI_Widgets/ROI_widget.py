from PyQt6.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QGridLayout,
    QDialog,
    QMessageBox,
)

from PyQt6.QtCore import Qt, QPoint, QEvent
import numpy as np
from shapely.geometry import Polygon
from cv2 import fitEllipse
from .list_layout import List_Layout
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from idtrackerai.utils.py_utils import (
    build_ROI_mask_from_list,
    get_vertices_from_label,
)


class ROI_Widget(List_Layout):
    def __init__(self, param_funcs):
        super().__init__()
        self.param_funcs = param_funcs
        self.CheckBox.setText("Region of interest")
        self.add.clicked.connect(self.add_clicked)

        self.ROI_popup = ROI_PopUp()
        self.WrongROI_PopUp = WrongROI_PopUp()

        self.list.itemClicked.connect(self.item_clicked)
        self.list.itemChanged.connect(self.item_clicked)

        self.ListChanged.connect(self.change_in_ROI_event)

        self.list.installEventFilter(self)
        self.ROI_mask = None

    def change_in_ROI_event(self):
        self.patches = build_ROI_patches_from_list(
            width=self.param_funcs["video_width"](),
            height=self.param_funcs["video_height"](),
            list_of_ROIs=self.str_list(),
        )

        self.ROI_mask = build_ROI_mask_from_list(
            width=self.param_funcs["video_width"](),
            height=self.param_funcs["video_height"](),
            list_of_ROIs=self.str_list(),
        )

    def eventFilter(self, object, event):
        if event.type() in (QEvent.WindowDeactivate, QEvent.FocusOut):
            self.plot_line.set_data([], [])
            self.list.clearSelection()
            self.draw_and_flush.emit(0)
        return False

    def item_clicked(self, item):
        if self.add.isChecked():
            return
        line = item.data(Qt.UserRole)
        self.plot_line.set_data(
            *self.get_vertices_from_label(line, close=True).T
        )
        self.plot_line.set(linestyle="-", marker=None)
        self.draw_and_flush.emit(0)

    def add_clicked(self, checked):
        if checked:
            if self.ROI_popup.exec(self.add):
                self.ROI_type = self.ROI_popup.value
                self.plot_line.set_data([], [])
                self.plot_line.set(linestyle="", marker=".")
                self.draw_and_flush.emit(0)
            else:
                self.add.setChecked(False)
        else:
            xy = self.plot_line.get_xydata().astype(np.int32)
            self.plot_line.set_data([], [])

            if self.ROI_type[2:9] == "Polygon":
                if len(xy) < 3:
                    self.WrongROI_PopUp.exec_with_message(
                        "Polygons can only be defined with 3 points or more"
                    )
                elif not Polygon(xy).is_valid:
                    self.WrongROI_PopUp.exec_with_message(
                        "Polygons can't intersect with themselves"
                    )
                else:
                    self.add_str_to_list(
                        f"{self.ROI_type} {[list(pair) for pair in xy]}"
                    )
            elif self.ROI_type[2:9] == "Ellipse":
                if len(xy) < 5:
                    self.WrongROI_PopUp.exec_with_message(
                        "Ellipses can only be defined with 5 points"
                        "(exact fit) or more (approximated fit)"
                    )
                else:
                    center, axis, angle = fitEllipse(xy)
                    axis = axis[0] / 2.0, axis[1] / 2.0
                    angle = 2 * np.pi * angle / 360
                    self.add_str_to_list(
                        f"{self.ROI_type} [{center[0]:.1f}, {center[1]:.1f},"
                        f" {axis[0]:.1f}, {axis[1]:.1f}, {angle:.3f}]"
                    )

    def get_patches(self):
        if self.CheckBox.isChecked():
            return self.patches
        else:
            return []

    def get_mask(self):
        if self.CheckBox.isChecked() and self.list.count():
            return self.ROI_mask
        else:
            return np.ones(
                (
                    self.param_funcs["video_height"](),
                    self.param_funcs["video_width"](),
                ),
                bool,
            )


def shapely_poly_to_mpl_patch(poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
    )
    return PathPatch(path, **kwargs)


def build_ROI_patches_from_list(width, height, list_of_ROIs):
    if not list_of_ROIs:
        return []
    else:
        main_poly = Polygon([[0, 0], [0, height], [width, height], [width, 0]])
        for line in list_of_ROIs:
            polygon = Polygon(get_vertices_from_label(line))

            if line[0] == "+":
                main_poly = main_poly.difference(polygon)
            elif line[0] == "-":
                main_poly = main_poly.union(polygon)
            else:
                raise TypeError

        if isinstance(main_poly, Polygon):
            return [shapely_poly_to_mpl_patch(main_poly, color="r", alpha=0.2)]
        else:
            # if it is not a Polygon, it is a collection of Polygons
            return [
                shapely_poly_to_mpl_patch(polygon, color="r", alpha=0.2)
                for polygon in main_poly.geoms
            ]


class ROI_PopUp(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(300, 100)
        self.setWindowTitle("Add ROI type")
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        PP_button = QPushButton("Positive Polygon")
        PE_button = QPushButton("Positive Ellipse")
        NP_button = QPushButton("Negative Polygon")
        NE_button = QPushButton("Negative Ellipse")

        PP_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        PE_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        NP_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        NE_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        PP_button.setStyleSheet("background-color: #60ff60")
        PE_button.setStyleSheet("background-color: #60ff60")
        NP_button.setStyleSheet("background-color: #ff6060")
        NE_button.setStyleSheet("background-color: #ff6060")

        def selected(value):
            self.value = value
            self.accept()

        PP_button.clicked.connect(lambda: selected("+ Polygon"))
        PE_button.clicked.connect(lambda: selected("+ Ellipse"))
        NP_button.clicked.connect(lambda: selected("- Polygon"))
        NE_button.clicked.connect(lambda: selected("- Ellipse"))

        grid.addWidget(PP_button, 0, 0)
        grid.addWidget(PE_button, 0, 1)
        grid.addWidget(NP_button, 1, 0)
        grid.addWidget(NE_button, 1, 1)

    def exec(self, trigger_widget):
        # Move the QDialog window to the widget that has called it
        point = trigger_widget.rect().bottomRight()
        global_point = trigger_widget.mapToGlobal(point)
        self.move(global_point - QPoint(self.width(), 0))
        # And run the QDialog
        return super().exec()


class WrongROI_PopUp(QMessageBox):
    def __init__(self):
        super().__init__()
        self.setText("Wrong ROI")
        self.setIcon(QMessageBox.Warning)
        self.setStandardButtons(QMessageBox.Ok)

    def exec_with_message(self, message):
        self.setInformativeText(message)
        return super().exec()
