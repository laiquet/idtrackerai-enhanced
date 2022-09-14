from PyQt6.QtWidgets import (
    QCheckBox,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QListWidget,
    QSizePolicy,
    QGridLayout,
    QDialog,
    QMessageBox,
)

from PyQt6.QtCore import Qt, QPoint
import numpy as np
from shapely.geometry import Polygon
from cv2 import fitEllipse


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

    def exec(self, callWidget):
        # Move the QDialog window to the widget that has called it
        point = callWidget.rect().bottomRight()
        global_point = callWidget.mapToGlobal(point)
        self.move(global_point - QPoint(self.width(), 0))
        # And run the QDialog
        return super().exec()


class ROI_Widget:
    def __init__(self):

        self.CheckBox = QCheckBox("Apply ROI", enabled=False)
        self.CheckBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.CheckBox.stateChanged.connect(self.CheckBox_changed)

        self.ROI_mode_isactive = False
        self.add_ROI = QPushButton("Add ROI", enabled=False)
        self.add_ROI.clicked.connect(self.add_ROI_func)
        self.remove_ROI = QPushButton("Remove selected", enabled=False)
        self.mask = None

        def remove_ROI_func():
            for item in self.ROI_list.selectedItems():
                self.ROI_list.takeItem(self.ROI_list.row(item))
            # self.share_updated_ROI()
            if not len(self.ROI_list.selectedItems()):
                self.remove_ROI.setEnabled(False)

        self.remove_ROI.clicked.connect(remove_ROI_func)
        self.ROI_list = QListWidget(visible=False)
        self.ROI_list.addItem("control_item")
        self.ROI_list.setFixedHeight(
            self.ROI_list.sizeHintForRow(0) * 5
            + 2 * self.ROI_list.frameWidth(),
        )
        self.ROI_list.clear()

        self.ROI_list.itemClicked.connect(
            lambda: self.remove_ROI.setEnabled(
                self.CheckBox.isChecked()
                and len(self.ROI_list.selectedItems())
            )
        )

        self.ROI_Layout = QVBoxLayout()
        ROI_Controls_HBox = QHBoxLayout()
        ROI_Controls_HBox.addWidget(self.CheckBox)
        ROI_Controls_HBox.addWidget(self.add_ROI)
        ROI_Controls_HBox.addWidget(self.remove_ROI)

        self.ROI_Layout.addLayout(ROI_Controls_HBox)
        self.ROI_Layout.addWidget(self.ROI_list)

        self.list_of_widgets = [self.CheckBox, self.ROI_list]
        self.ROI_popup = ROI_PopUp()

        self.WrongROI_PopUp = QMessageBox()
        self.WrongROI_PopUp.setText("Wrong ROI")
        self.WrongROI_PopUp.setIcon(QMessageBox.Warning)
        self.WrongROI_PopUp.setStandardButtons(QMessageBox.Ok)

    # def setEnabled(self, enabled):
    #     self.CheckBox.setEnabled(enabled)
    #     self.ROI_list.setEnabled(enabled)
    #     self.remove_ROI.setEnabled(
    #         self.CheckBox.isChecked() and len(self.ROI_list.selectedItems())
    #     )
    #     self.add_ROI.setEnabled(self.CheckBox.isChecked())

    def CheckBox_changed(self, state):
        self.ROI_list.setVisible(state)
        self.add_ROI.setEnabled(state)
        self.remove_ROI.setEnabled(
            state and len(self.ROI_list.selectedItems())
        )
        # self.share_updated_ROI()

    def add_ROI_func(self):
        if self.ROI_popup.exec(self.add_ROI):
            self.ROI_type = self.ROI_popup.value
            self.ROI_mode_isactive = True

    def setROIEnabled(self, enabled):
        self.CheckBox.setEnabled(enabled)
        self.ROI_list.setEnabled(enabled)
        self.remove_ROI.setEnabled(
            self.CheckBox.isChecked() and len(self.ROI_list.selectedItems())
        )
        self.add_ROI.setEnabled(self.CheckBox.isChecked())
        # print("setting ROI to", enabled)

    def enter_key_event(self):
        if not self.ROI_mode_isactive:
            return
        xy = self.building_ROI.get_xydata().astype(np.int32)
        self.building_ROI.set_data([], [])
        self.ROI_mode_isactive = False

        if self.ROI_type[2:9] == "Polygon":
            if len(xy) < 3:
                self.CheckBox.stateChanged.emit(1)
                self.WrongROI_PopUp.setInformativeText(
                    "Polygons can only be defined with 3 points or more"
                )
                self.WrongROI_PopUp.exec()
            elif not Polygon(xy).is_valid:
                self.CheckBox.stateChanged.emit(1)
                self.WrongROI_PopUp.setInformativeText(
                    "Polygons can't intersect with themselves"
                )
                self.WrongROI_PopUp.exec()
            else:
                self.ROI_list.addItem(
                    f"{self.ROI_type} {[list(pair) for pair in xy]}"
                )
        elif self.ROI_type[2:9] == "Ellipse":
            if len(xy) < 5:
                self.CheckBox.stateChanged.emit(1)
                self.WrongROI_PopUp.setInformativeText(
                    "Ellipses can only be defined with 5 points (exact fit) or more (approximated fit)"
                )
                self.WrongROI_PopUp.exec()
            else:
                center, axis, angle = fitEllipse(xy)
                axis = axis[0] / 2.0, axis[1] / 2.0
                angle = 2 * np.pi * angle / 360
                self.ROI_list.addItem(
                    f"{self.ROI_type} [{center[0]:.1f}, {center[1]:.1f}, {axis[0]:.1f}, {axis[1]:.1f}, {angle:.3f}]"
                )

    def click_event(self, event):
        xy = self.building_ROI.get_xydata()
        self.building_ROI.set_data(
            np.vstack([xy, (event.xdata, event.ydata)]).T
        )
        # print(f'recieved click {event.x = } {event.y = }')

    # def end_ROI_mode(self):
    #     xy = self.building_ROI.get_xydata().astype(int)
    #     # str_data = [list(pair) for pair in xy]
    #     self.building_ROI.set_data([], [])
    #     self.ROI_mode_isactive = False
    #     self.ROI_list.addItem(f"{self.ROI_type} {[list(pair) for pair in xy]}")
    #     return self.str_ROI_list

    @property
    def str_ROI_list(self) -> str:
        if self.CheckBox.isChecked():
            return "\n".join(
                self.ROI_list.item(i).text()
                for i in range(self.ROI_list.count())
            )
        else:
            return None
