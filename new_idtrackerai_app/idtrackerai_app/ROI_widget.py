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
import json
import cv2
from matplotlib.path import Path
from matplotlib.patches import PathPatch


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

        self.list.model().rowsInserted.connect(self.build_ROI_from_QListWidget)
        self.list.model().rowsRemoved.connect(self.build_ROI_from_QListWidget)
        self.CheckBox.stateChanged.connect(self.build_ROI_from_QListWidget)
        self.list.installEventFilter(self)
        self.ROI_mask = None

    def eventFilter(self, object, event):
        if event.type() in (QEvent.WindowDeactivate, QEvent.FocusOut):
            self.plot_line.set_data([], [])
            self.list.clearSelection()
            self.draw_and_flush()
        return False

    def item_clicked(self, item):
        if self.add.isChecked():
            return
        line = item.data(Qt.UserRole)
        self.plot_line.set_data(
            *self.get_vertices_from_label(line, close=True).T
        )
        self.plot_line.set(linestyle="-", marker=None)
        self.draw_and_flush()

    def add_clicked(self, checked):
        if checked:
            if self.ROI_popup.exec(self.add):
                self.ROI_type = self.ROI_popup.value
                self.plot_line.set_data([], [])
                self.plot_line.set(linestyle="", marker=".")
                self.draw_and_flush()
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
                        "Ellipses can only be defined with 5 points (exact fit) or more (approximated fit)"
                    )
                else:
                    center, axis, angle = fitEllipse(xy)
                    axis = axis[0] / 2.0, axis[1] / 2.0
                    angle = 2 * np.pi * angle / 360
                    self.add_str_to_list(
                        f"{self.ROI_type} [{center[0]:.1f}, {center[1]:.1f}, {axis[0]:.1f}, {axis[1]:.1f}, {angle:.3f}]"
                    )

    @staticmethod
    def get_vertices_from_label(label: str, close=False):
        if label[2:9] == "Polygon":
            vertices = np.asarray(json.loads(label[10:]))
        elif label[2:9] == "Ellipse":
            ox, oy, a, b, angle = json.loads(label[10:])
            t = np.linspace(0, 2 * np.pi, 100)
            x = a * np.cos(t)
            y = b * np.sin(t)
            rot_x = np.cos(angle) * x - np.sin(angle) * y + ox
            rot_y = np.sin(angle) * x + np.cos(angle) * y + oy
            vertices = np.asarray([rot_x, rot_y]).T
        else:
            raise TypeError

        if close:
            return np.vstack([vertices, vertices[0]]).astype(np.int32)
        else:
            return vertices.astype(np.int32)

    def build_ROI_from_QListWidget(self):
        width = self.param_funcs["video_width"]()
        height = self.param_funcs["video_height"]()
        list_of_ROIs = self.str_list()

        if list_of_ROIs is None:
            self.patches = []
            self.ROI_mask = np.ones((height, width), np.uint8)
        else:

            self.ROI_mask = np.zeros((height, width), np.uint8)
            main_poly = Polygon(
                [[0, 0], [0, height], [width, height], [width, 0]]
            )
            for line in list_of_ROIs.splitlines():

                vertices = self.get_vertices_from_label(line)
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
                self.patches = [
                    shapely_poly_to_mpl_patch(main_poly, color="r", alpha=0.2)
                ]
            else:
                # if it is not a Polygon, it is a collection of Polygons
                self.patches = [
                    shapely_poly_to_mpl_patch(polygon, color="r", alpha=0.2)
                    for polygon in main_poly.geoms
                ]
        self.update_mask_patches_on_VideoPlayer(self.patches)

    def get_patches(self):
        if self.CheckBox.isChecked():
            return self.patches
        else:
            return []

    def get_mask(self):
        if self.CheckBox.isChecked():
            if self.list.count():
                return self.ROI_mask
            else:
                return np.zeros(
                    (
                        self.param_funcs["video_height"](),
                        self.param_funcs["video_width"](),
                    ),
                    bool,
                )
        else:
            return np.ones(
                (
                    self.param_funcs["video_height"](),
                    self.param_funcs["video_width"](),
                ),
                bool,
            )


# Plots a Polygon to pyplot `ax`
def shapely_poly_to_mpl_patch(poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
    )
    return PathPatch(path, **kwargs)
