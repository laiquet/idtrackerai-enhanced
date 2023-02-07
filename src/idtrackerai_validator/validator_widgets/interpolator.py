import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
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
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = WrappedLabel()
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.interpolation_type_box = CustomComboBox()
        self.interpolation_type_box.addItems(self.interpolation_kinds.keys())
        self.interpolation_type_box.setCurrentText("cubic")
        self.interpolation_type_box.currentTextChanged.connect(self.new_interp_type)
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Interpolation order"))
        order_row.addWidget(self.interpolation_type_box)
        layout.addLayout(order_row)

        radio_row = QHBoxLayout()
        radio_row.addWidget(QLabel("Input size"))

        for value in (10, 150, 1500):
            btn = QRadioButton(str(value))
            if value == 10:
                btn.setChecked(True)
            btn.clicked.connect(self.new_input_size)
            radio_row.addWidget(btn)

        layout.addLayout(radio_row)
        self.input_size = 10

        apply_row = QHBoxLayout()
        style = self.style()
        cancel_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogCancelButton), "Cancel"
        )
        cancel_btn.setShortcut(Qt.Key.Key_Escape)
        cancel_btn.clicked.connect(lambda: self.setActivated(False))

        apply_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogOkButton),
            "Apply interpolation",
        )
        apply_btn.setShortcut(Qt.Key.Key_Return)

        apply_btn.clicked.connect(self.apply_interpolation)
        apply_row.addWidget(cancel_btn)
        apply_row.addWidget(apply_btn)
        layout.addLayout(apply_row)

        self.setActivated(False)

    def new_interp_type(self, type: str):
        self.interp1d = interp1d(
            self.interp1d.x,
            self.interp1d.y,
            kind=self.interpolation_kinds[type],  # type: ignore
            fill_value="extrapolate",  # type: ignore
            assume_sorted=True,
        )
        self.neew_to_draw.emit()

    def new_input_size(self):
        btn = self.sender()
        assert isinstance(btn, QRadioButton)
        self.input_size = int(btn.text())
        self.build_interpolator()

    def set_interpolation_params(self, id, start, end):
        self.start = start
        self.end = end
        self.id = id - 1

        self.interpolation_range = range(self.start, self.end)
        self.continuous_interpolation_range = np.arange(
            self.start - 1, self.end + 0.1, 0.2
        )
        self.build_interpolator()

    def build_interpolator(self):
        time_range = np.arange(
            max(0, self.start - self.input_size),
            min(self.n_frames, self.end + self.input_size),
        )

        time_range = time_range[~np.isnan(self.trajectories[time_range, self.id, 0])]

        self.interp1d = interp1d(
            time_range,
            self.trajectories[time_range, self.id].T,
            kind=self.interpolation_kinds[
                self.interpolation_type_box.currentText()
            ],  # type:ignore
            fill_value="extrapolate",  # type:ignore
            assume_sorted=True,
        )
        self.title.setText(
            f"Interpolation for id {self.id+1}\nfrom frame {self.start} to {self.end}"
        )
        self.setActivated(True)

    def redirect_keyReleaseEvent(self, event: QKeyEvent):
        if not self.isEnabled():
            return
        if event.key() == Qt.Key.Key_R:
            ...

    def setActivated(self, activated: bool):
        self.setEnabled(activated)
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
        x_input = self.interp1d.x
        y_input = self.interp1d.y.T

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
        painter.drawPolyline([QPointF(*xy) for xy in y_input[x_input < self.start]])  # type: ignore
        painter.drawPolyline([QPointF(*xy) for xy in y_input[x_input >= self.end]])  # type: ignore
        for point in y_input:
            painter.drawBigPoint(*point)

        # actual point
        painter.setPenColor(0xFFFFFF)
        painter.drawBigPoint(*self.interp1d(frame))
