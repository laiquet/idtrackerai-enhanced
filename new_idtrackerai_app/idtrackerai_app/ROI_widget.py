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
    def __init__(self):
        super().__init__()
        self.CheckBox.setText("Region of interest")
        self.add.clicked.connect(self.add_clicked)

        self.ROI_popup = ROI_PopUp()
        self.WrongROI_PopUp = WrongROI_PopUp()

        self.list.itemActivated.connect(self.item_clicked)

        self.list.installEventFilter(self)

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
