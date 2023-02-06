import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import interp1d

from idtrackerai import ListOfBlobs
from idtrackerai_GUI_tools import CustomPainter, WrappedLabel, key_event_modifier


class CustomComboBox(QComboBox):
    def keyPressEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyReleaseEvent(event)


class Interpolator(QWidget):
    interpolation_kinds = {"linear": 1, "quadratic": 2, "cubic": 3, "5th order": 5}
    neew_to_draw = pyqtSignal()
    update_trajectories = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.activated = False
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = WrappedLabel()
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.interpolation_type_box = CustomComboBox()
        self.interpolation_type_box.addItems(self.interpolation_kinds.keys())
        self.interpolation_type_box.setCurrentText("cubic")
        self.interpolation_type_box.currentTextChanged.connect(self.new_interp_type)
        self.order_row = QHBoxLayout()
        self.order_label = QLabel("Interpolation order")
        self.order_row.addWidget(self.order_label)
        self.order_row.addWidget(self.interpolation_type_box)
        layout.addLayout(self.order_row)

        apply_row = QHBoxLayout()
        self.cancel_btn = QToolButton()
        self.cancel_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.cancel_btn.setText("Cancel")
        self.cancel_btn.setShortcut(Qt.Key.Key_Escape)
        self.cancel_btn.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_DialogCancelButton)
        )
        self.cancel_btn.clicked.connect(lambda: self.setActivated(False))

        self.apply_btn = QToolButton()
        self.apply_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.apply_btn.setText("Apply interpolation")
        self.apply_btn.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_DialogOkButton)
        )

        self.apply_btn.clicked.connect(self.apply_interpolation)
        apply_row.addWidget(self.cancel_btn)
        apply_row.addWidget(self.apply_btn)
        layout.addLayout(apply_row)

        self.pad_extra = 0
        self.pad = 10

    def new_interp_type(self, type: str):
        self.interp1d = interp1d(
            self.interp1d.x,
            self.interp1d.y,
            kind=self.interpolation_kinds[type],
            fill_value="extrapolate",
            assume_sorted=True,
        )
        self.neew_to_draw.emit()

    def set_interpolation_params(self, id, start, end):
        self.start = start
        self.end = end
        self.id = id - 1

        self.interpolation_range = range(self.start, self.end)
        self.continuous_interpolation_range = np.arange(
            self.start - 1, self.end + 0.1, 0.2
        )

        time_range = np.arange(
            max(0, self.start - (self.pad + self.pad_extra)),
            min(self.n_frames, self.end + (self.pad + self.pad_extra)),
        )

        time_range = time_range[~np.isnan(self.trajectories[time_range, self.id, 0])]

        self.interp1d = interp1d(
            time_range,
            self.trajectories[time_range, self.id].T,
            kind=self.interpolation_kinds[self.interpolation_type_box.currentText()],
            fill_value="extrapolate",
            assume_sorted=True,
        )
        self.title.setText(
            f"Interpolation for id {id}\nfrom frame {self.start} to {self.end}"
        )
        self.setActivated(True)

    def redirect_keyReleaseEvent(self, key: Qt.Key):
        if not self.activated:
            return
        if key == Qt.Key.Key_R:
            ...

    def setActivated(self, activated: bool):
        self.activated = activated
        self.cancel_btn.setEnabled(activated)
        self.apply_btn.setEnabled(activated)
        self.order_label.setEnabled(activated)
        self.interpolation_type_box.setEnabled(activated)
        if not activated:
            self.title.setText(
                'Select some errors of kind "Miss id" of '
                '"Jump" to start an interpolation process'
            )
        self.neew_to_draw.emit()

    def apply_interpolation(self):
        for new_centroid, frame in zip(
            self.interp1d(self.interpolation_range).T, self.interpolation_range
        ):
            if not np.isnan(self.trajectories[frame, self.id, 0]):
                continue

            contains_centroid = [
                blob.contains_point(new_centroid)
                for blob in self.list_of_blobs.blobs_in_video[frame]
            ]
            if any(contains_centroid):
                blob = self.list_of_blobs.blobs_in_video[frame][
                    contains_centroid.index(True)
                ]
            else:
                blob = min(
                    self.list_of_blobs.blobs_in_video[frame],
                    key=lambda b: b.distance_from_countour_to(new_centroid),
                )
            blob.add_centroid(new_centroid, self.id + 1)
        self.update_trajectories.emit(self.start, self.end)
        self.neew_to_draw.emit()

    def set_references(
        self,
        traj: np.ndarray,
        all_identified: np.ndarray,
        duplicated: np.ndarray,
        list_of_blobs: ListOfBlobs,
    ):
        self.list_of_blobs = list_of_blobs
        self.trajectories = traj
        self.all_identified = all_identified
        self.duplicated = duplicated
        self.n_frames = self.trajectories.shape[0]

    def paint_on_canvas(self, painter: CustomPainter, frame: int):
        # interpolated points
        painter.setPenColor(0xFFFFFF)
        for point in self.interp1d(self.interpolation_range).T:
            painter.drawBigPoint(*point)

        # continuum interpolated range
        painter.drawPolyline(
            [
                QPointF(*xy)
                for xy in self.interp1d(self.continuous_interpolation_range).T
            ]  # type: ignore
        )

        # interpolator input data
        painter.setPenColor(0xFF0000)
        painter.setBrush(0xFF0000)
        for point in self.interp1d.y.T:
            painter.drawBigPoint(*point)

        # actual point
        painter.setPenColor(0xFFFFFF)
        painter.drawBigPoint(*self.interp1d(frame))
