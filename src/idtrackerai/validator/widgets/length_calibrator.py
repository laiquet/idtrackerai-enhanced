from qtpy.QtCore import QPointF, Qt, Signal  # type: ignore
from qtpy.QtGui import QColor, QColorConstants
from qtpy.QtWidgets import QInputDialog, QToolButton, QVBoxLayout, QWidget

from idtrackerai.GUI_tools import (
    CanvasMouseEvent,
    CanvasPainter,
    CustomList,
    LightPopUp,
    point_colors,
)
from idtrackerai.utils import LengthCalibration


class LengthCalibrator(QWidget):
    needToDraw = Signal()
    current_calibration: LengthCalibration | None = None

    def __init__(self) -> None:
        super().__init__()

        self.add = QToolButton()
        self.add.setText("Add")

        self.list = CustomList(max_n_row=0)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        self.setLayout(layout)
        layout.addWidget(self.add)
        layout.addWidget(self.list)

        self.add.clicked.connect(self.add_clicked)
        self.calibrations: list[LengthCalibration] = []
        self.list.ListChanged.connect(self.needToDraw.emit)
        self.list.removedItem.connect(self.remove_item)
        self.color_count = -1
        self.popup = LightPopUp()
        self.info_dialog_already_displayed: bool = False

    def click_event(self, event: CanvasMouseEvent) -> None:
        if (
            not self.isVisible()
            or not self.isEnabled()
            or self.current_calibration is None
        ):
            return

        if event.button == Qt.MouseButton.LeftButton:
            self.current_calibration.add_point(event.int_xy_data)
            if self.current_calibration.has_two_points():
                self.end_calibration()

    def end_calibration(self) -> None:
        if self.current_calibration is None:
            return

        valid = False
        dialog_text = "Enter real distance between the two points:"
        while not valid:
            value_str, ok = QInputDialog.getText(self.add, "idtracker.ai", dialog_text)
            if not ok:
                self.current_calibration = None
                return
            try:
                value = float(value_str)
            except ValueError:
                valid = False
            else:
                valid = True
            if not valid:
                dialog_text = "Invalid characters encountered"

        self.current_calibration.distance = value
        self.calibrations.append(self.current_calibration)
        self.list.add_str(
            str(self.current_calibration), color=QColor(self.current_calibration.color)
        )
        self.current_calibration = None

    def add_clicked(self) -> None:
        if not self.info_dialog_already_displayed:
            self.popup.info(
                "Length Calibration",
                "Indicate two points by clicking in the video and then enter the numeric value of the distance between these two. The average factor of all calibrations will be included in the trajectory files.",
            )
            self.info_dialog_already_displayed = True

        self.color_count = (
            0 if self.color_count == len(point_colors) - 1 else self.color_count + 1
        )
        self.current_calibration = LengthCalibration(point_colors[self.color_count])
        self.needToDraw.emit()

    def remove_item(self, data: str) -> None:
        for calibration in self.calibrations:
            if str(calibration) == data:
                self.calibrations.remove(calibration)
                return

    def load(self, calibrations: list[LengthCalibration] | None) -> None:
        self.list.clear()
        if calibrations is None:
            return

        for calibration in calibrations:
            self.color_count = (
                0 if self.color_count == len(point_colors) - 1 else self.color_count + 1
            )
            calibration.color = point_colors[self.color_count]
            self.calibrations.append(calibration)
            self.list.add_str(str(calibration), color=QColor(calibration.color))
        self.current_calibration = None

    def get_calibrations(self) -> list[LengthCalibration]:
        return [c for c in self.calibrations if c.completed()]

    def paint_on_canvas(self, painter: CanvasPainter) -> None:
        painter.setPenColor(QColorConstants.Black)
        for calibration in self.calibrations:
            assert calibration.point_A is not None
            assert calibration.point_B is not None
            painter.setPenColor(QColor(calibration.color))
            painter.drawLine(*(calibration.point_A + calibration.point_B))  # type: ignore
            painter.setPenColor(QColorConstants.Black)
            painter.drawText(
                QPointF(
                    0.5 * (calibration.point_A[0] + calibration.point_B[0]),
                    0.5 * (calibration.point_A[1] + calibration.point_B[1]),
                ),
                f"{calibration.distance}",
            )
            painter.setBrush(QColor(calibration.color))
            painter.drawBigPoint(*calibration.point_A)
            painter.drawBigPoint(*calibration.point_B)

        calibration = self.current_calibration
        if calibration is None:
            return
        painter.setBrush(QColor(calibration.color))

        if calibration.point_A is not None:
            painter.drawBigPoint(*calibration.point_A)

        if calibration.point_B is not None:
            painter.drawBigPoint(*calibration.point_B)
