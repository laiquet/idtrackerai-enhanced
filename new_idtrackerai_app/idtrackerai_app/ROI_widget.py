from PyQt6.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QGridLayout,
    QDialog,
    QMessageBox,
)

from PyQt6.QtCore import Qt, QPoint
import numpy as np
from shapely.geometry import Polygon
from cv2 import fitEllipse
from .list_layout import List_Layout


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
        self.CheckBox.setText("ROI")
        self.add.clicked.connect(self.add_clicked)

        self.ROI_popup = ROI_PopUp()
        self.WrongROI_PopUp = WrongROI_PopUp()

    def add_clicked(self, checked):
        if checked:
            if self.ROI_popup.exec(self.add):
                self.ROI_type = self.ROI_popup.value
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
                    self.list.addItem(
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
                    self.list.addItem(
                        f"{self.ROI_type} [{center[0]:.1f}, {center[1]:.1f}, {axis[0]:.1f}, {axis[1]:.1f}, {angle:.3f}]"
                    )
