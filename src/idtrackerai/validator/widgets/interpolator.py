import logging

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
    QStyle,
    QToolButton,
    QVBoxLayout,
)
from scipy.interpolate import BSpline, make_interp_spline

from idtrackerai import ListOfBlobs
from idtrackerai.GUI_tools import (
    CanvasMouseEvent,
    CanvasPainter,
    LightPopUp,
    WrappedLabel,
    key_event_modifier,
)


class CustomComboBox(QComboBox):
    def keyPressEvent(self, e: QKeyEvent) -> None:
        event = key_event_modifier(e)
        if event is not None:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, e: QKeyEvent) -> None:
        event = key_event_modifier(e)
        if event is not None:
            super().keyReleaseEvent(event)


class Interpolator(QGroupBox):
    interpolation_kinds = {"Linear": 1, "Quadratic": 2, "Cubic": 3, "5th order": 5}
    neew_to_draw = Signal()
    update_trajectories = Signal(int, int, bool)  # start, end, update_errors
    go_to_frame = Signal(int)
    preload_frames = Signal(int, int)
    interpolation_accepted = Signal()
    enabled_changed = Signal(bool)
    interp_spline: BSpline

    def __init__(self) -> None:
        self.popup = LightPopUp()
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
        self.interpolation_order_box.setCurrentText("Cubic")
        self.interpolation_order_box.currentTextChanged.connect(self.new_interp_type)
        order_row = QHBoxLayout()
        self.interpolation_order_label = WrappedLabel("Spline order")
        order_row.addWidget(self.interpolation_order_label)
        order_row.addWidget(self.interpolation_order_box)
        layout.addLayout(order_row)

        self.input_size_row = QHBoxLayout()
        self.input_size_row.addWidget(WrappedLabel("Real values"))
        for value in (10, 100, 1000):
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
        assert style is not None

        self.abort_btn = QPushButton(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton),
            "Abort [Esc]",
        )
        self.abort_btn.setShortcut(Qt.Key.Key_Escape)
        self.abort_btn.clicked.connect(self.abort_interpolation)
        apply_row.addWidget(self.abort_btn)

        self.apply_btn = QPushButton(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOkButton),
            "Apply [Ctrl+A]",
        )
        self.apply_btn.setShortcut("Ctrl+A")
        self.apply_btn.clicked.connect(self.apply_interpolation)
        apply_row.addWidget(self.apply_btn)

        layout.addLayout(apply_row)

        self.setActivated(False)
        self.animal_id: int = -1

    def trajectories_have_been_updated(self) -> None:
        if self.isEnabled():
            self.build_interpolator()

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.EnabledChange:
            self.enabled_changed.emit(self.isEnabled())

    def new_interp_type(self, kind: str) -> None:
        self.interp_spline = make_interp_spline(
            self.interp_x,
            self.interp_y,
            k=self.interpolation_kinds[kind],
            check_finite=False,
        )
        self.neew_to_draw.emit()

    def new_input_size(self) -> None:
        btn = self.sender()
        assert isinstance(btn, QRadioButton)
        self.input_size = int(btn.text())
        self.build_interpolator()

    def set_interpolation_params(self, animal_id: int, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.animal_id = animal_id - 1
        self.expand_start()
        self.expand_end()
        self.preload_frames.emit(max(0, self.start - 10), self.start)
        self.build_interpolator()

    def build_interpolator(self) -> None:
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
        try:
            self.interp_x = times_were_not_nan
            self.interp_y = self.trajectories[times_were_not_nan, self.animal_id]
            self.interp_spline = make_interp_spline(
                self.interp_x,
                self.interp_y,
                k=self.interpolation_kinds[self.interpolation_order_box.currentText()],
                check_finite=False,
            )
        except ValueError as exc:
            self.setActivated(False)
            logging.error("Unexpected error", exc_info=exc)
            QMessageBox.warning(
                self,
                "Unexpected error",
                f"Unexpected error while initializing interpolation:\n{exc}",
            )
        else:
            self.setActivated(True)
            self.info_label.setText(
                "Interpolating identity <span"
                f' style="font-weight:600">{self.animal_id+1}'
            )

    def remove_current_centroid(self) -> None:
        if self.current_frame not in self.entire_range:
            QMessageBox.warning(
                self,
                "Interpolator message",
                "Cannot remove current centroid outside interpolation "
                f"range ({self.entire_range.start} -> {self.entire_range.stop})",
            )
            return

        centroid_to_remove = self.trajectories[self.current_frame, self.animal_id]
        if np.isnan(centroid_to_remove[0]):
            self.popup.warning(
                "Interpolator message",
                "Cannot remove current centroid because it does not exist",
            )
            return

        self.list_of_blobs.remove_centroid(
            self.current_frame, centroid_to_remove, self.animal_id + 1
        )
        self.update_trajectories.emit(self.current_frame, self.current_frame + 1, False)

        self.expand_end()
        self.expand_start()

        self.build_interpolator()

    def expand_start(self) -> None:
        for frame in range(self.start - 1, -1, -1):
            if not np.isnan(self.trajectories[frame, self.animal_id, 0]):
                if frame + 1 != self.start:
                    self.start = frame + 1
                    self.go_to_frame.emit(frame)
                return

    def expand_end(self) -> None:
        for frame in range(self.end, self.n_frames):
            if not np.isnan(self.trajectories[frame, self.animal_id, 0]):
                if frame != self.end:
                    self.end = frame
                    self.go_to_frame.emit(frame)
                return

    def click_event(self, event: CanvasMouseEvent) -> None:
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
        self.update_trajectories.emit(self.current_frame, self.current_frame + 1, False)

    def setActivated(self, activated: bool) -> None:
        self.setEnabled(activated)
        if not activated:
            self.warning.setVisible(False)
            self.goto_btn.setVisible(False)
            self.info_label.setText(
                'Select some errors of kind "Miss id" of '
                '"Jump" to start an interpolation process'
            )
        self.neew_to_draw.emit()

    def abort_interpolation(self) -> None:
        logging.debug("Abort interpolation")
        self.update_trajectories.emit(self.start, self.end, True)
        self.setActivated(False)

    def apply_interpolation(self) -> None:
        logging.debug("Apply interpolation")
        for new_centroid, frame in zip(
            self.interp_spline(self.interpolation_range), self.interpolation_range
        ):
            if np.isnan(self.trajectories[frame, self.animal_id, 0]):
                self.list_of_blobs.add_centroid(frame, self.animal_id + 1, new_centroid)
        self.setActivated(False)
        self.interpolation_accepted.emit()
        self.update_trajectories.emit(self.start, self.end, True)

    @property
    def start(self) -> int:
        return self._start

    @start.setter
    def start(self, value: int) -> None:
        self._start = value
        self.start_btn.setText(f"frame {value-1}")
        self.start_btn.setToolTip(f"Go to frame {value-1}")

    @property
    def end(self) -> int:
        return self._end

    @end.setter
    def end(self, value: int) -> None:
        self._end = value
        self.end_btn.setText(f"frame {value}")
        self.end_btn.setToolTip(f"Go to frame {value}")

    def set_references(
        self,
        traj: np.ndarray,
        unidentified: np.ndarray,
        duplicated: np.ndarray,
        list_of_blobs: ListOfBlobs,
    ) -> None:
        self.list_of_blobs = list_of_blobs
        self.trajectories = traj
        self.unidentified = unidentified
        self.duplicated = duplicated
        self.n_frames = len(self.trajectories)

    def paint_on_canvas(self, painter: CanvasPainter, frame: int) -> None:
        self.current_frame = frame

        # interpolated points
        painter.setPenColor(QColorConstants.White)
        painter.setBrush(QColorConstants.White)
        for point in self.interp_spline(self.interpolation_range):
            painter.drawBigPoint(*point)

        # continuum interpolated range
        painter.drawPolyline([
            QPointF(*xy)
            for xy in self.interp_spline(self.continuous_interpolation_range)
        ])  # type: ignore

        # interpolator input data
        painter.setPenColor(QColorConstants.Red)
        painter.setBrush(QColorConstants.Red)
        painter.drawPolyline([QPointF(*xy) for xy in self.interp_y[self.interp_x < self.start]])  # type: ignore
        painter.drawPolyline([QPointF(*xy) for xy in self.interp_y[self.interp_x >= self.end]])  # type: ignore
        for point in self.interp_y:
            painter.drawBigPoint(*point)

        # actual point
        if (
            self.current_frame in self.interp_x
            or self.current_frame in self.interpolation_range
        ):
            painter.setPenColor(QColorConstants.White)
            painter.drawBigPoint(*self.interp_spline(frame))
