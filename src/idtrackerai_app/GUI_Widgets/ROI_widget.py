import numpy as np
from cv2 import fitEllipse
from idtrackerai_app.widgets_utils import CustomQPainter, ListLayout, MessageBox
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPolygonF
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
)

from idtrackerai.utils import build_ROI_mask_from_list, get_vertices_from_label


class ROIWidget(ListLayout):
    def __init__(self, parent):
        super().__init__(name="Region of interest", parent=parent)
        self.add.clicked.connect(self.add_clicked)
        self.ListChanged.connect(self.update_Patches)
        self.ListChanged.connect(lambda: self.valueChanged.emit(self.getMask()))

        self.ROI_popup = ROI_PopUp(parent)
        self.WrongROI_PopUp = MessageBox(parent, "Wrong ROI")
        self.newItemSelected.connect(self.paint_selected_polygon)
        self.mask_path = QPainterPath()
        self.clicked_points = []
        self.ListItem_clicked = False

    def click_event(self, button, x, y):
        if self.add.isChecked():
            self.clicked_points.append((x, y))
            self.needToDraw.emit()

    def paint_selected_polygon(self, new: QListWidgetItem):
        if new:
            self.ListItem_clicked = True
            line = new.data(Qt.ItemDataRole.UserRole)
            self.clicked_points = list(
                map(tuple, get_vertices_from_label(line, close=True))
            )

        else:
            self.ListItem_clicked = False
            self.clicked_points.clear()
        self.needToDraw.emit()

    def add_clicked(self, checked):
        if checked:
            if self.ROI_popup.exec():
                self.ROI_type = self.ROI_popup.value
                self.needToDraw.emit()
            else:
                self.add.setChecked(False)
        else:
            xy = self.clicked_points
            self.needToDraw.emit()

            if self.ROI_type[2:9] == "Polygon":
                if len(xy) < 3:
                    self.WrongROI_PopUp.exec(
                        message="Polygons can only be defined with 3 points or more"
                    )
                else:
                    self.add_str_to_list(
                        f"{self.ROI_type} ["
                        + ", ".join([f"[{x:.1f}, {y:.1f}]" for x, y in xy])
                        + "]"
                    )
            elif self.ROI_type[2:9] == "Ellipse":
                if len(xy) < 5:
                    self.WrongROI_PopUp.exec(
                        message="Ellipses can only be defined with 5 points"
                        "(exact fit) or more (approximated fit)"
                    )
                else:
                    center, axis, angle = fitEllipse(np.asarray(xy, dtype=np.float32))
                    axis = axis[0] / 2.0, axis[1] / 2.0
                    angle = 2 * np.pi * angle / 360
                    self.add_str_to_list(
                        f"{self.ROI_type} "
                        + "{"
                        + f"'center': [{center[0]:.1f}, {center[1]:.1f}], "
                        f"'axes': [{axis[0]:.1f}, {axis[1]:.1f}], 'angle': {angle:.3f}"
                        + "}"
                    )
        self.clicked_points.clear()

    def set_video_size(self, video_size):
        self.video_size = video_size

    def update_Patches(self):
        if self.CheckBox.isChecked():
            self.mask_path = build_ROI_patches_from_list(
                *self.video_size, list_of_ROIs=self.getValue()
            )
        else:
            self.mask_path = QPainterPath()

    def getMask(self):
        if self.CheckBox.isChecked():
            return build_ROI_mask_from_list(
                *self.video_size, list_of_ROIs=self.getValue()
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

    def paint_on_canvas(self, painter: CustomQPainter):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 0, 0, 50))
        painter.drawPath(self.mask_path)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPenColor(QColor(50, 100, 10))
        if self.ListItem_clicked:
            painter.drawPolygonFromVertices(self.clicked_points)

        painter.setBrush(QColor(50, 150, 80))
        for point in self.clicked_points:
            painter.drawBigPoint(*point)


def build_ROI_patches_from_list(width, height, list_of_ROIs) -> QPainterPath:
    path = QPainterPath()
    if not list_of_ROIs:
        return path
    else:
        path = QPainterPath()
        path.addRect(0, 0, width, height)

        for line in list_of_ROIs:
            points = get_vertices_from_label(line)
            path_i = QPainterPath(QPointF(*points[0]))
            for point in points[1:]:
                path_i.lineTo(*point)

            if line[0] == "+":
                path -= path_i.simplified()
            elif line[0] == "-":
                path += path_i.simplified()
            else:
                raise TypeError
        return path


class ROI_PopUp(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setFixedSize(300, 100)
        self.setWindowTitle("Add ROI type")
        layout = QGridLayout()
        self.setLayout(layout)

        PP_button = QPushButton("+ Polygon")
        PE_button = QPushButton("+ Ellipse")
        NP_button = QPushButton("- Polygon")
        NE_button = QPushButton("- Ellipse")

        policy = QSizePolicy.Policy.Expanding
        PP_button.setSizePolicy(policy, policy)
        PE_button.setSizePolicy(policy, policy)
        NP_button.setSizePolicy(policy, policy)
        NE_button.setSizePolicy(policy, policy)

        PP_button.clicked.connect(self.clicked_event)
        PE_button.clicked.connect(self.clicked_event)
        NP_button.clicked.connect(self.clicked_event)
        NE_button.clicked.connect(self.clicked_event)

        layout.addWidget(PP_button, 0, 0)
        layout.addWidget(PE_button, 0, 1)
        layout.addWidget(NP_button, 1, 0)
        layout.addWidget(NE_button, 1, 1)

    def clicked_event(self):
        self.value = self.sender().text()
        self.accept()
