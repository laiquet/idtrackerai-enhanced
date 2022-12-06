import numpy as np
from cv2 import fitEllipse
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QGridLayout, QPushButton, QSizePolicy
from shapely.geometry import Polygon

from idtrackerai.utils.py_utils import (
    build_ROI_mask_from_list,
    get_vertices_from_label,
)
from idtrackerai_app.widgets_utils import ListLayout, MessageBox


class ROIWidget(ListLayout):
    def __init__(self, parent, ax: Axes):
        super().__init__(name="Region of interest", parent=parent)
        self.add.clicked.connect(self.add_clicked)
        self.ListChanged.connect(self.update_Patches)
        self.ListChanged.connect(
            lambda: self.valueChanged.emit(self.getMask())
        )

        self.ROI_popup = ROI_PopUp(parent)
        self.WrongROI_PopUp = MessageBox(parent, "Wrong ROI")
        self.newItemSelected.connect(self.paint_selected_polygon)
        self.mask_polygons = []
        self.ax = ax
        self.Line2D = self.ax.plot([], [], ".", animated=True)[0]

    def click_event(self, button, x, y):
        if self.add.isChecked():
            xy = self.Line2D.get_xydata()
            self.Line2D.set_data(np.vstack([xy, (x, y)]).T)
            self.needToDraw.emit()

    def paint_selected_polygon(self, new):
        if new:
            line = new.data(Qt.UserRole)
            self.Line2D.set_data(*get_vertices_from_label(line, close=True).T)
            self.Line2D.set(linestyle="-", marker=None)
        else:
            self.Line2D.set_data([], [])
        self.needToDraw.emit()

    def add_clicked(self, checked):
        if checked:
            if self.ROI_popup.exec():
                self.ROI_type = self.ROI_popup.value
                self.Line2D.set(linestyle="", marker=".", data=([], []))
                self.needToDraw.emit()
            else:
                self.add.setChecked(False)
        else:
            xy = self.Line2D.get_xydata().astype(np.int32)
            self.Line2D.set_data([], [])
            self.needToDraw.emit()

            if self.ROI_type[2:9] == "Polygon":
                if len(xy) < 3:
                    self.WrongROI_PopUp.exec(
                        message="Polygons can only be defined with 3 points or more"
                    )
                elif not Polygon(xy).is_valid:
                    self.WrongROI_PopUp.exec(
                        message="Polygons can't intersect with themselves"
                    )
                else:
                    self.add_str_to_list(
                        f"{self.ROI_type} {[list(pair) for pair in xy]}"
                    )
            elif self.ROI_type[2:9] == "Ellipse":
                if len(xy) < 5:
                    self.WrongROI_PopUp.exec(
                        message="Ellipses can only be defined with 5 points"
                        "(exact fit) or more (approximated fit)"
                    )
                else:
                    center, axis, angle = fitEllipse(xy)
                    axis = axis[0] / 2.0, axis[1] / 2.0
                    angle = 2 * np.pi * angle / 360
                    self.add_str_to_list(
                        f"{self.ROI_type} "
                        + "{"
                        + f"'center': [{center[0]:.1f}, {center[1]:.1f}], "
                        f"'axes': [{axis[0]:.1f}, {axis[1]:.1f}], 'angle': {angle:.3f}"
                        + "}"
                    )

    def set_video_size(self, video_size):
        self.video_size = video_size

    def update_Patches(self):
        while self.mask_polygons:
            self.mask_polygons.pop().remove()

        if self.CheckBox.isChecked():
            self.mask_polygons = build_ROI_patches_from_list(
                *self.video_size,
                list_of_ROIs=self.getValue(),
            )
        else:
            self.mask_polygons = []

        for polygon in self.mask_polygons:
            self.ax.add_patch(polygon)

    def getMask(self):
        if self.CheckBox.isChecked():
            return build_ROI_mask_from_list(
                *self.video_size,
                list_of_ROIs=self.getValue(),
            )
        else:
            return np.ones(self.video_size[::-1], bool)

    def setValue(self, values: list[str]):
        if not values:
            return
        if isinstance(values, str):
            values = [values]
        for value in values:
            self.add_str_to_list(value)
        self.CheckBox.setChecked(True)

    def draw_artists(self, renderer):
        self.Line2D.draw(renderer)
        for mask_polygon in self.mask_polygons:
            mask_polygon.draw(renderer)


def shapely_poly_to_mpl_patch(poly: Polygon, **kwargs) -> PathPatch:
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
    )
    return PathPatch(path, **kwargs)


def build_ROI_patches_from_list(
    width, height, list_of_ROIs
) -> list[PathPatch]:
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
            return [
                shapely_poly_to_mpl_patch(
                    main_poly, color="r", alpha=0.2, animated=True
                )
            ]
        else:
            # if it is not a Polygon, it is a collection of Polygons
            return [
                shapely_poly_to_mpl_patch(
                    polygon, color="r", alpha=0.2, animated=True
                )
                for polygon in main_poly.geoms
            ]


class ROI_PopUp(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(300, 100)
        self.setWindowTitle("Add ROI type")
        self.setLayout(QGridLayout())

        PP_button = QPushButton("+ Polygon")
        PE_button = QPushButton("+ Ellipse")
        NP_button = QPushButton("- Polygon")
        NE_button = QPushButton("- Ellipse")

        PP_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        PE_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        NP_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        NE_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        PP_button.clicked.connect(self.clicked_event)
        PE_button.clicked.connect(self.clicked_event)
        NP_button.clicked.connect(self.clicked_event)
        NE_button.clicked.connect(self.clicked_event)

        self.layout().addWidget(PP_button, 0, 0)
        self.layout().addWidget(PE_button, 0, 1)
        self.layout().addWidget(NP_button, 1, 0)
        self.layout().addWidget(NE_button, 1, 1)

    def clicked_event(self):
        self.value = self.sender().text()
        self.accept()
