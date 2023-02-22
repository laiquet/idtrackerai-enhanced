from math import sqrt

import numpy as np
from cv2 import fitEllipse
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainterPath
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.utils import build_ROI_mask_from_list, get_vertices_from_label
from idtrackerai_GUI_tools import CustomList, CustomPainter, build_ROI_patches_from_list


class ROIWidget(QWidget):
    needToDraw = pyqtSignal()
    valueChanged = pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super().__init__()

        self.CheckBox = QCheckBox("Regions of interest")
        self.CheckBox.stateChanged.connect(self.CheckBox_changed)

        self.add = QToolButton()
        self.add.setText("Add")
        self.add.setCheckable(True)
        self.add.setVisible(False)

        self.list = CustomList()
        self.list.setVisible(False)

        # self.list.ListChanged.connect(self.list.update_height)

        Controls_HBox = QHBoxLayout()
        Controls_HBox.addWidget(self.CheckBox)
        Controls_HBox.addWidget(self.add)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        self.setLayout(layout)
        layout.addLayout(Controls_HBox)
        layout.addWidget(self.list)

        self.add.toggled.connect(self.add_clicked)
        self.list.ListChanged.connect(self.update_Patches)
        self.list.ListChanged.connect(lambda: self.valueChanged.emit(self.getMask()))

        self.ROI_popup = ROI_PopUp(parent)
        self.list.newItemSelected.connect(self.paint_selected_polygon)
        self.mask_path = QPainterPath()
        self.clicked_points = []
        self.ListItem_clicked = False

    def getValue(self) -> list[str]:
        return self.list.getValue() if self.CheckBox.isChecked() else []

    def getMask(self) -> np.ndarray:
        return build_ROI_mask_from_list(*self.video_size, list_of_ROIs=self.getValue())

    def CheckBox_changed(self, enabled):
        if self.add.isChecked():
            self.add.click()
        self.list.setVisible(enabled)
        self.add.setVisible(enabled)
        self.valueChanged.emit(self.getMask())

    def click_event(self, button: int, zoom: float, x: float, y: float):
        if self.add.isChecked():
            if button == Qt.MouseButton.LeftButton:
                # Add clicked point
                self.clicked_points.append((x, y))
            elif button == Qt.MouseButton.RightButton:
                # Remove nearest point
                if not self.clicked_points:
                    return
                distances = [
                    sqrt((px - x) ** 2 + (py - y) ** 2)
                    for px, py in self.clicked_points
                ]
                index, dist = min(enumerate(distances), key=lambda x: x[1])
                if dist < 20 * zoom:  # 20 px threshold
                    self.clicked_points.pop(index)
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
            if not xy:  # any drawn points
                return
            self.needToDraw.emit()

            if self.ROI_type[2:9] == "Polygon":
                if len(xy) < 3:
                    QMessageBox.warning(
                        self,
                        "ROI error",
                        "Polygons can only be defined with 3 points or more",
                    )
                else:
                    self.list.add_str(
                        f"{self.ROI_type} ["
                        + ", ".join([f"[{x:.1f}, {y:.1f}]" for x, y in xy])
                        + "]"
                    )
            elif self.ROI_type[2:9] == "Ellipse":
                if len(xy) < 5:
                    QMessageBox.warning(
                        self,
                        "ROI error",
                        "Ellipses can only be defined with 5 points"
                        "(exact fit) or more (approximated fit)",
                    )
                else:
                    center, axis, angle = fitEllipse(np.asarray(xy, dtype="f"))
                    axis = axis[0] / 2.0, axis[1] / 2.0
                    self.list.add_str(
                        f"{self.ROI_type} "
                        + "{"
                        + f"'center': [{center[0]:.0f}, {center[1]:.0f}], "
                        f"'axes': [{axis[0]:.0f}, {axis[1]:.0f}], 'angle': {angle:.0f}"
                        + "}"
                    )
        self.clicked_points.clear()

    def set_video_size(self, video_size):
        self.video_size = video_size

    def update_Patches(self):
        self.mask_path = build_ROI_patches_from_list(
            *self.video_size, list_of_ROIs=self.getValue()
        )

    def setValue(self, values: list[str]):
        if not values:
            return
        if isinstance(values, str):
            values = [values]
        self.CheckBox.setChecked(True)
        for value in values:
            self.list.add_str(value)

    def paint_on_canvas(self, painter: CustomPainter):
        if not self.CheckBox.isChecked():
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 0, 0, 50))
        painter.drawPath(self.mask_path)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPenColor(0x32640A)
        if self.ListItem_clicked:
            painter.drawPolygonFromVertices(self.clicked_points)
        else:
            painter.setBrush(0x349650)
            for point in self.clicked_points:
                painter.drawBigPoint(*point)

    def enter_key_event(self):
        self.add.setChecked(False)


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
        sender = self.sender()
        assert isinstance(sender, QPushButton)
        self.value = sender.text()
        self.accept()
