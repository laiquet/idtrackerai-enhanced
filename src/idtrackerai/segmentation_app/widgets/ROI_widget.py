from itertools import cycle

import numpy as np
from cv2 import fitEllipse
from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtGui import QColor, QPainterPath
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QListView,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.base.fragmentation.fragmentation import find_exclusive_contours
from idtrackerai.GUI_tools import (
    CanvasMouseEvent,
    CanvasPainter,
    CustomList,
    build_ROI_patches_from_list,
    get_path_from_points,
)
from idtrackerai.utils import build_ROI_mask_from_list, get_vertices_from_label


class ROIWidget(QWidget):
    needToDraw = Signal()
    valueChanged = Signal(object)  # np.ndarray | None
    exclusive_ROI_colors = [
        QColor(0, 0, 190, 200),
        QColor(0, 200, 0, 200),
        QColor(255, 100, 0, 200),
        QColor(205, 254, 0, 200),
        QColor(138, 102, 66, 200),
        QColor(200, 50, 250, 200),
    ]
    clicked_points: list[tuple[float, float]]

    def __init__(self, parent):
        super().__init__()

        self.CheckBox = QCheckBox("Regions of interest")

        self.add = QToolButton()
        self.add.setText("Add")
        self.add.setCheckable(True)
        self.add.setEnabled(False)

        self.list = CustomList()
        self.list.setEnabled(False)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setMovement(QListView.Movement.Free)

        self.exclusive_rois = QCheckBox("Exclusive ROIs")
        self.exclusive_rois.setEnabled(False)

        Controls_HBox = QHBoxLayout()
        Controls_HBox.addWidget(self.CheckBox)
        Controls_HBox.addWidget(self.add)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        self.setLayout(layout)
        layout.addLayout(Controls_HBox)
        layout.addWidget(self.list)
        layout.addWidget(self.exclusive_rois, alignment=Qt.AlignmentFlag.AlignRight)

        self.add.toggled.connect(self.add_clicked)
        self.list.ListChanged.connect(self.update_ROI)
        self.CheckBox.stateChanged.connect(self.CheckBox_changed)
        self.CheckBox.stateChanged.connect(self.update_ROI)
        self.exclusive_rois.stateChanged.connect(self.update_ROI)

        self.ROI_popup = ROI_PopUp(parent)
        self.list.newItemSelected.connect(self.paint_selected_polygon)
        self.mask_path = QPainterPath()
        self.clicked_points = []
        self.ListItem_clicked = False
        self.resolution_reduction = 1
        self.video_size = 1, 1

    def getValue(self) -> list[str] | None:
        return self.list.getValue() if self.CheckBox.isChecked() else None

    def CheckBox_changed(self, enabled):
        if self.add.isChecked():
            self.add.click()
        self.list.setEnabled(enabled)
        self.add.setEnabled(enabled)
        self.exclusive_rois.setEnabled(enabled)

    def click_event(self, event: CanvasMouseEvent):
        if self.add.isChecked():
            if event.button == Qt.MouseButton.LeftButton:
                # Add clicked point
                self.clicked_points.append(event.xy_data)
            elif event.button == Qt.MouseButton.RightButton:
                # Remove nearest point
                if not self.clicked_points:
                    return
                distances = map(event.distance_to, self.clicked_points)
                index, dist = min(enumerate(distances), key=lambda x: x[1])
                if dist < 20 * event.zoom:  # 20 px threshold
                    self.clicked_points.pop(index)
            self.needToDraw.emit()

    def paint_selected_polygon(self, new: QListWidgetItem):
        if new:
            self.ListItem_clicked = True
            line = new.data(Qt.ItemDataRole.UserRole)
            self.clicked_points = list(
                map(tuple, get_vertices_from_label(line, close=True).astype(np.int32))
            )  # type: ignore

        else:
            self.ListItem_clicked = False
            self.clicked_points.clear()
        self.needToDraw.emit()

    def add_clicked(self, checked):
        xy, self.clicked_points = self.clicked_points, []

        if checked:
            self.ROI_type = self.ROI_popup.exec(self.list.count() == 0)
            if self.ROI_type is None:
                self.add.setChecked(False)
            return

        if not xy:  # any drawn points
            return

        assert self.ROI_type is not None

        xy = np.asarray(xy) / self.resolution_reduction

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

    def set_resolution_reduction(self, resolution_reduction: float):
        self.resolution_reduction = resolution_reduction
        self.update_ROI()

    def set_video_size(self, video_size):
        self.video_size = video_size

    def update_ROI(self):
        list_of_ROIs = self.getValue()
        self.mask_path = build_ROI_patches_from_list(
            list_of_ROIs, self.resolution_reduction, *self.video_size
        )

        mask = build_ROI_mask_from_list(
            list_of_ROIs, self.resolution_reduction, *self.video_size
        )

        self.exclusive_ROI_paths: list[QPainterPath] = []
        for contour, holes in find_exclusive_contours(mask):
            path = get_path_from_points(contour)
            for hole in holes:
                path -= get_path_from_points(hole)
            self.exclusive_ROI_paths.append(path)

        if len(self.exclusive_ROI_paths) == 1:
            self.exclusive_ROI_paths.clear()

        self.exclusive_rois.setVisible(len(self.exclusive_ROI_paths) > 0)

        self.valueChanged.emit(mask)

    def setValue(self, values: list[str] | str | None, exclusive_roi: bool):
        self.list.clear()
        if not values:
            return
        if isinstance(values, str):
            values = [values]
        self.CheckBox.setChecked(True)
        for value in values:
            self.list.add_str(value)
        self.exclusive_rois.setChecked(exclusive_roi)

    def paint_on_canvas(self, painter: CanvasPainter):
        if not self.CheckBox.isChecked():
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 0, 0, 50))
        painter.drawPath(self.mask_path)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.exclusive_rois.isChecked():
            for color, path in zip(
                cycle(self.exclusive_ROI_colors), self.exclusive_ROI_paths
            ):
                painter.setPenColor(color)
                painter.drawPath(path)

        painter.setPenColor(QColor(0x32640A))
        if self.ListItem_clicked:
            painter.drawPolygonFromVertices(
                self.clicked_points, self.resolution_reduction
            )
        else:
            painter.setBrush(QColor(0x349650))
            for point in self.clicked_points:
                painter.drawBigPoint(*point)


class ROI_PopUp(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setFixedSize(300, 100)
        self.setWindowTitle("Add ROI type")
        layout = QGridLayout()
        self.setLayout(layout)

        self.PP_button = QPushButton("+ Polygon")
        self.PE_button = QPushButton("+ Ellipse")
        self.NP_button = QPushButton("- Polygon")
        self.NE_button = QPushButton("- Ellipse")

        policy = QSizePolicy.Policy.Expanding
        self.PP_button.setSizePolicy(policy, policy)
        self.PE_button.setSizePolicy(policy, policy)
        self.NP_button.setSizePolicy(policy, policy)
        self.NE_button.setSizePolicy(policy, policy)

        self.PP_button.clicked.connect(self.clicked_event)
        self.PE_button.clicked.connect(self.clicked_event)
        self.NP_button.clicked.connect(self.clicked_event)
        self.NE_button.clicked.connect(self.clicked_event)

        layout.addWidget(self.PP_button, 0, 0)
        layout.addWidget(self.PE_button, 0, 1)
        layout.addWidget(self.NP_button, 1, 0)
        layout.addWidget(self.NE_button, 1, 1)

    def clicked_event(self):
        sender = self.sender()
        assert isinstance(sender, QPushButton)
        self.value = sender.text()
        self.accept()

    def exec(self, only_positive: bool = False) -> str | None:
        self.NP_button.setEnabled(not only_positive)
        self.NE_button.setEnabled(not only_positive)

        return self.value if super().exec() == QDialog.DialogCode.Accepted else None
