import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
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
    raise_warning = pyqtSignal(str)
    go_to_frame = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.warning = WrappedLabel()
        self.warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.warning)
        self.warning.setVisible(False)

        self.goto_btn = QPushButton()
        layout.addWidget(self.goto_btn)
        self.goto_btn.setVisible(False)

        self.title = WrappedLabel()
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.interpolation_type_box = CustomComboBox()
        self.interpolation_type_box.addItems(self.interpolation_kinds.keys())
        self.interpolation_type_box.setCurrentText("cubic")
        self.interpolation_type_box.currentTextChanged.connect(self.new_interp_type)
        order_row = QHBoxLayout()
        order_row.addWidget(WrappedLabel("Interpolation order"))
        order_row.addWidget(self.interpolation_type_box)
        layout.addLayout(order_row)

        radio_row = QHBoxLayout()
        radio_row.addWidget(WrappedLabel("Input size"))
        for value in (10, 150, 1500):
            btn = QRadioButton(str(value))
            if value == 10:
                btn.setChecked(True)
            btn.clicked.connect(self.new_input_size)
            radio_row.addWidget(btn)
        layout.addLayout(radio_row)
        self.input_size = 10

        remove_centroid = QPushButton("Remove centroid [R]")
        remove_centroid.setShortcut(Qt.Key.Key_R)
        remove_centroid.clicked.connect(self.remove_current_centroid)
        layout.addWidget(remove_centroid)

        apply_row = QHBoxLayout()
        style = self.style()
        cancel_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogCancelButton),
            "Cancel [Esc]",
        )
        cancel_btn.setShortcut(Qt.Key.Key_Escape)
        cancel_btn.clicked.connect(lambda: self.setActivated(False))
        apply_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogOkButton),
            "Interpolate [I]",
        )
        apply_btn.setShortcut(Qt.Key.Key_I)
        apply_btn.clicked.connect(self.apply_interpolation)
        apply_row.addWidget(cancel_btn)
        apply_row.addWidget(apply_btn)
        layout.addLayout(apply_row)

        self.setActivated(False)

    def trajectories_have_been_updated(self):
        if self.isEnabled():
            self.build_interpolator()

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
        self.build_interpolator()

    def build_interpolator(self):
        self.interpolation_range = range(self.start, self.end)
        self.continuous_interpolation_range = np.arange(
            self.start - 1, self.end + 0.1, 0.2
        )
        self.entire_range = range(
            max(0, self.start - self.input_size),
            min(self.n_frames, self.end + self.input_size),
        )

        n_duplicated = np.count_nonzero(self.duplicated[self.entire_range, self.id])
        if n_duplicated:
            first_duplicated = (
                self.duplicated[self.entire_range, self.id].argmax()
                + self.entire_range.start
            )
            self.warning.setText(
                '<font color="red">'
                f"There are {n_duplicated} frames where identity {self.id+1} appears "
                "duplicated. It is highly recommended to solve this before proceeding. "
                f"The first duplication appears at frame {first_duplicated}"
            )
            self.goto_btn.setText(f"Go to {first_duplicated}")
            self.goto_btn.clicked.connect(
                lambda: self.go_to_frame.emit(first_duplicated)
            )
            self.warning.setVisible(True)
            self.goto_btn.setVisible(True)
        else:
            self.warning.setVisible(False)
            self.goto_btn.setVisible(False)

        times_were_not_nan = np.asarray(self.entire_range)[
            ~np.isnan(self.trajectories[self.entire_range, self.id, 0])
        ]

        self.interp1d = interp1d(
            times_were_not_nan,
            self.trajectories[times_were_not_nan, self.id].T,
            kind=self.interpolation_kinds[
                self.interpolation_type_box.currentText()
            ],  # type:ignore
            fill_value="extrapolate",  # type:ignore
            assume_sorted=True,
        )
        self.title.setText(
            f"Interpolation for id {self.id+1} from frame {self.start} to {self.end}"
        )
        self.setActivated(True)

    def remove_current_centroid(self):
        if self.current_frame not in self.entire_range:
            return self.raise_warning.emit(
                "Cannot remove current centroid outside interpolation "
                f"range ({self.entire_range.start} -> {self.entire_range.stop})"
            )

        centroid_to_remove = self.trajectories[self.current_frame, self.id]
        id_to_remove = self.id + 1
        if np.isnan(centroid_to_remove[0]):
            return self.raise_warning.emit(
                "Cannot remove current centroid because it does not exist"
            )

        for blob in self.list_of_blobs.blobs_in_video[self.current_frame]:
            blob.remove_centroid(id_to_remove, centroid_to_remove)

        self.update_trajectories.emit(self.current_frame, self.current_frame + 1)

        if self.current_frame == self.start - 1:
            for frame in range(self.start, -1, -1):
                if not np.isnan(self.trajectories[frame, self.id, 0]):
                    self.start = frame + 1
                    self.go_to_frame.emit(frame)
                    break
        elif self.current_frame == self.end:
            for frame in range(self.end, self.n_frames):
                if not np.isnan(self.trajectories[frame, self.id, 0]):
                    self.end = frame
                    self.go_to_frame.emit(frame)
                    break

        self.build_interpolator()

    def click_event(self, button: int, zoom: float, x: float, y: float):
        if not self.isEnabled() or self.current_frame not in self.interpolation_range:
            return

        current_postion = self.trajectories[self.current_frame, self.id]
        already_has_a_centroid = not np.isnan(current_postion[0])
        if already_has_a_centroid:
            self.list_of_blobs.update_centroid(
                self.current_frame, self.id + 1, current_postion, (x, y)
            )
        else:
            self.list_of_blobs.add_centroid(self.current_frame, self.id + 1, (x, y))
        self.update_trajectories.emit(self.current_frame, self.current_frame + 1)

    def setActivated(self, activated: bool):
        self.setEnabled(activated)
        if not activated:
            self.warning.setVisible(False)
            self.goto_btn.setVisible(False)
            self.title.setText(
                'Select some errors of kind "Miss id" of '
                '"Jump" to start an interpolation process'
            )
        self.neew_to_draw.emit()

    def apply_interpolation(self):
        for new_centroid, frame in zip(
            self.interp1d(self.interpolation_range).T, self.interpolation_range
        ):
            if np.isnan(self.trajectories[frame, self.id, 0]):
                self.list_of_blobs.add_centroid(frame, self.id + 1, new_centroid)
        self.setEnabled(False)
        self.update_trajectories.emit(self.start, self.end)

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
        self.current_frame = frame
        x_input = self.interp1d.x
        y_input = self.interp1d.y.T

        # interpolated points
        painter.setPenColor(0xFFFFFF)
        painter.setBrush(0xFFFFFF)
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
        if self.current_frame in self.entire_range:
            painter.setPenColor(0xFFFFFF)
            painter.drawBigPoint(*self.interp1d(frame))
