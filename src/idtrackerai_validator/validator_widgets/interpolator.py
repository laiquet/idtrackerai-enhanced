import numpy as np
from qtpy.QtCore import QEvent, QPointF, Qt, Signal  # type: ignore
from qtpy.QtGui import QColorConstants, QKeyEvent
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
)
from scipy.interpolate import interp1d

from idtrackerai import ListOfBlobs
from idtrackerai_GUI_tools import (
    CanvasMouseEvent,
    CanvasPainter,
    WrappedLabel,
    key_event_modifier,
)


class CustomComboBox(QComboBox):
    def keyPressEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyReleaseEvent(event)


class Interpolator(QGroupBox):
    interpolation_kinds = {"linear": 1, "quadratic": 2, "cubic": 3, "5th order": 5}
    neew_to_draw = Signal()
    update_trajectories = Signal(int, int)
    go_to_frame = Signal(int)
    preload_frames = Signal(int, int)
    interpolation_accepted = Signal()
    enabled_changed = Signal(bool)

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

        self.info_label = WrappedLabel()
        layout.addWidget(self.info_label)

        range_row = QHBoxLayout()
        range_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.start_btn = QToolButton()
        self.end_btn = QToolButton()
        self.start_btn.clicked.connect(lambda: self.go_to_frame.emit(self.start - 1))
        self.end_btn.clicked.connect(lambda: self.go_to_frame.emit(self.end))
        range_row.addWidget(QLabel("From"))
        range_row.addWidget(self.start_btn)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.end_btn)
        layout.addLayout(range_row)

        self.interpolation_order_box = CustomComboBox()
        self.interpolation_order_box.addItems(self.interpolation_kinds.keys())
        self.interpolation_order_box.setCurrentText("cubic")
        self.interpolation_order_box.currentTextChanged.connect(self.new_interp_type)
        order_row = QHBoxLayout()
        self.interpolation_order_label = WrappedLabel("Interpolation order")
        order_row.addWidget(self.interpolation_order_label)
        order_row.addWidget(self.interpolation_order_box)
        layout.addLayout(order_row)

        self.input_size_row = QHBoxLayout()
        self.input_size_row.addWidget(WrappedLabel("Input size"))
        for value in (10, 150, 1500):
            btn = QRadioButton(str(value))
            if value == 10:
                btn.setChecked(True)
            btn.clicked.connect(self.new_input_size)
            self.input_size_row.addWidget(btn)
        layout.addLayout(self.input_size_row)
        self.input_size = 10

        remove_centroid = QPushButton("Remove centroid [R]")
        remove_centroid.setShortcut(Qt.Key.Key_R)
        remove_centroid.clicked.connect(self.remove_current_centroid)
        layout.addWidget(remove_centroid)

        apply_row = QHBoxLayout()
        style = self.style()
        self.abort_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogCancelButton),
            "Abort [Esc]",
        )
        self.apply_btn = QPushButton(
            style.standardIcon(style.StandardPixmap.SP_DialogOkButton), "Apply [Ctrl+A]"
        )
        self.abort_btn.setShortcut(Qt.Key.Key_Escape)
        self.apply_btn.setShortcut("Ctrl+A")
        self.abort_btn.clicked.connect(lambda: self.setActivated(False))
        self.apply_btn.clicked.connect(self.apply_interpolation)
        apply_row.addWidget(self.abort_btn)
        apply_row.addWidget(self.apply_btn)
        layout.addLayout(apply_row)

        self.setActivated(False)
        self.animal_id: int = -1
        self.interp1d: interp1d
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def trajectories_have_been_updated(self):
        if self.isEnabled():
            self.build_interpolator()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.EnabledChange:
            self.enabled_changed.emit(self.isEnabled())

    def new_interp_type(self, kind: str):
        self.interp1d = interp1d(
            self.interp1d.x,
            self.interp1d.y,
            kind=self.interpolation_kinds[kind],  # type: ignore
            fill_value="extrapolate",  # type: ignore
            assume_sorted=True,
        )
        self.neew_to_draw.emit()

    def new_input_size(self):
        btn = self.sender()
        assert isinstance(btn, QRadioButton)
        self.input_size = int(btn.text())
        self.build_interpolator()

    def set_interpolation_params(self, animal_id, start, end):
        self.start = start
        self.end = end
        self.animal_id = animal_id - 1
        self.expand_start()
        self.expand_end()
        self.preload_frames.emit(max(0, self.start - 10), self.start)
        self.build_interpolator()

    def build_interpolator(self):
        self.interpolation_range = range(self.start, self.end)
        self.continuous_interpolation_range = np.arange(
            max(self.start - 1, 0), self.end + 0.1, 0.2
        )
        self.entire_range = range(
            max(0, self.start - self.input_size),
            min(self.n_frames, self.end + self.input_size),
        )

        n_duplicated = np.count_nonzero(
            self.duplicated[self.entire_range, self.animal_id]
        )
        if n_duplicated:
            first_duplicated = (
                self.duplicated[self.entire_range, self.animal_id].argmax()
                + self.entire_range.start
            )
            self.warning.setText(
                f'<font color="red">There are {n_duplicated} frames where identity'
                f" {self.animal_id+1} appears duplicated. It is highly recommended to"
                " solve this before proceeding. The first duplication appears at frame"
                f" {first_duplicated}"
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
            ~np.isnan(self.trajectories[self.entire_range, self.animal_id, 0])
        ]

        self.interp1d = interp1d(
            times_were_not_nan,
            self.trajectories[times_were_not_nan, self.animal_id].T,
            kind=self.interpolation_kinds[
                self.interpolation_order_box.currentText()
            ],  # type:ignore
            fill_value="extrapolate",  # type:ignore
            assume_sorted=True,
        )
        self.info_label.setText(
            f'Interpolating identity <span style="font-weight:600">{self.animal_id+1}'
        )
        self.setActivated(True)
        self.setFocus()

    def remove_current_centroid(self):
        if self.current_frame not in self.entire_range:
            QMessageBox.warning(
                self,
                "Interpolator message",
                (
                    "Cannot remove current centroid outside interpolation "
                    f"range ({self.entire_range.start} -> {self.entire_range.stop})"
                ),
            )
            return

        centroid_to_remove = self.trajectories[self.current_frame, self.animal_id]
        if np.isnan(centroid_to_remove[0]):
            QMessageBox.warning(
                self,
                "Interpolator message",
                "Cannot remove current centroid because it does not exist",
            )
            return

        self.list_of_blobs.remove_centroid(
            self.current_frame, centroid_to_remove, self.animal_id + 1
        )
        self.update_trajectories.emit(self.current_frame, self.current_frame + 1)

        if self.current_frame == self.start - 1:
            self.expand_start()
            self.go_to_frame.emit(self.start - 1)
        elif self.current_frame == self.end:
            self.expand_end()
            self.go_to_frame.emit(self.end)

        self.build_interpolator()

    def expand_start(self):
        for frame in range(self.start - 1, -1, -1):
            if not np.isnan(self.trajectories[frame, self.animal_id, 0]):
                self.start = frame + 1
                break

    def expand_end(self):
        for frame in range(self.end, self.n_frames):
            if not np.isnan(self.trajectories[frame, self.animal_id, 0]):
                self.end = frame
                break

    def click_event(self, event: CanvasMouseEvent):
        if (
            event.button != Qt.MouseButton.RightButton
            or not self.isEnabled()
            or self.current_frame not in self.interpolation_range
        ):
            return

        current_postion = self.trajectories[self.current_frame, self.animal_id]
        already_has_a_centroid = not np.isnan(current_postion[0])
        if already_has_a_centroid:
            self.list_of_blobs.update_centroid(
                self.current_frame, self.animal_id + 1, current_postion, event.xy_data
            )
        else:
            self.list_of_blobs.add_centroid(
                self.current_frame, self.animal_id + 1, event.xy_data
            )
        self.update_trajectories.emit(self.current_frame, self.current_frame + 1)

    def setActivated(self, activated: bool):
        self.setEnabled(activated)
        if not activated:
            self.warning.setVisible(False)
            self.goto_btn.setVisible(False)
            self.info_label.setText(
                'Select some errors of kind "Miss id" of '
                '"Jump" to start an interpolation process'
            )
        self.neew_to_draw.emit()

    def apply_interpolation(self):
        for new_centroid, frame in zip(
            self.interp1d(self.interpolation_range).T, self.interpolation_range
        ):
            if np.isnan(self.trajectories[frame, self.animal_id, 0]):
                self.list_of_blobs.add_centroid(frame, self.animal_id + 1, new_centroid)
        self.setEnabled(False)
        self.interpolation_accepted.emit()
        self.update_trajectories.emit(self.start, self.end)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._start = value
        self.start_btn.setText(f"frame {value-1}")
        self.start_btn.setToolTip(f"Go to frame {value-1}")

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = value
        self.end_btn.setText(f"frame {value}")
        self.end_btn.setToolTip(f"Go to frame {value}")

    def set_references(
        self,
        traj: np.ndarray,
        unidentified: np.ndarray,
        duplicated: np.ndarray,
        list_of_blobs: ListOfBlobs,
    ):
        self.list_of_blobs = list_of_blobs
        self.trajectories = traj
        self.unidentified = unidentified
        self.duplicated = duplicated
        self.n_frames = self.trajectories.shape[0]

    def paint_on_canvas(self, painter: CanvasPainter, frame: int):
        self.current_frame = frame
        x_input = self.interp1d.x
        y_input = self.interp1d.y.T

        # interpolated points
        painter.setPenColor(QColorConstants.White)
        painter.setBrush(QColorConstants.White)
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
        painter.setPenColor(QColorConstants.Red)
        painter.setBrush(QColorConstants.Red)
        painter.drawPolyline([QPointF(*xy) for xy in y_input[x_input < self.start]])  # type: ignore
        painter.drawPolyline([QPointF(*xy) for xy in y_input[x_input >= self.end]])  # type: ignore
        for point in y_input:
            painter.drawBigPoint(*point)

        # actual point
        if (
            self.current_frame in self.interp1d.x
            or self.current_frame in self.interpolation_range
        ):
            painter.setPenColor(QColorConstants.White)
            painter.drawBigPoint(*self.interp1d(frame))
