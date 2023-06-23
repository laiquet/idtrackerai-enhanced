from collections.abc import Iterable

from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import (
    QHBoxLayout,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from idtrackerai import Blob, Fragment


class MarkBlobs(QScrollArea):
    needToDraw = Signal()

    def __init__(self, parent: QWidget):
        main_layout = QVBoxLayout()
        super().__init__(parent)
        self.setWidgetResizable(True)
        wid = QWidget()
        wid.setLayout(main_layout)
        self.setWidget(wid)

        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.individual = QRadioButton("Individuals")
        self.crossing = QRadioButton("Crossings")
        self.used_for_training_crossings = QRadioButton("Used for\ntraining crossings")
        self.used_for_training = QRadioButton("Used for\ntraining identification")
        self.accumulable = QRadioButton("Accumulable")
        self.not_accumulated = QRadioButton("Accumulable but\nnot accumulated")
        self.accumulated = QRadioButton("Accumulated at")
        self.accumulation_spinbox = QSpinBox()
        self.accumulation_spinbox.setSpecialValueText("any step")
        self.accumulation_spinbox.setRange(-1, 1000)
        self.accumulation_spinbox.setValue(-1)
        self.accumulation_spinbox.setPrefix("step ")
        self.forces_to_crossing = QRadioButton("Forced to be crossing")

        self.individual.toggled.connect(self.needToDraw.emit)
        self.used_for_training.toggled.connect(self.needToDraw.emit)
        self.used_for_training_crossings.toggled.connect(self.needToDraw.emit)
        self.accumulable.toggled.connect(self.needToDraw.emit)
        self.accumulated.toggled.connect(self.needToDraw.emit)
        self.accumulation_spinbox.valueChanged.connect(self.needToDraw.emit)
        self.accumulation_spinbox.valueChanged.connect(
            lambda: self.accumulated.setChecked(True)
        )
        self.not_accumulated.toggled.connect(self.needToDraw.emit)
        self.forces_to_crossing.toggled.connect(self.needToDraw.emit)

        main_layout.addWidget(self.individual)
        main_layout.addWidget(self.used_for_training)
        main_layout.addWidget(self.used_for_training_crossings)
        main_layout.addWidget(self.accumulable)
        accumulated_at = QHBoxLayout()
        accumulated_at.setAlignment(Qt.AlignmentFlag.AlignLeft)
        accumulated_at.addWidget(self.accumulated)
        accumulated_at.addWidget(self.accumulation_spinbox)
        main_layout.addLayout(accumulated_at)
        main_layout.addWidget(self.not_accumulated)
        main_layout.addWidget(self.forces_to_crossing)

    def __call__(
        self, blobs: list[Blob], fragments: list[Fragment] | None
    ) -> Iterable[Blob]:
        if not self.isVisible() or not self.isEnabled():
            return ()

        if self.individual.isChecked():
            return (blob for blob in blobs if blob.is_an_individual)

        if self.crossing.isChecked():
            return (blob for blob in blobs if blob.is_a_crossing)

        if self.used_for_training_crossings.isChecked():
            return (blob for blob in blobs if blob.used_for_training_crossings)

        if fragments is None:
            return ()

        if self.used_for_training.isChecked():
            return filter(
                lambda blob: fragments[blob.fragment_identifier].used_for_training,
                blobs,
            )

        if self.accumulable.isChecked():
            return filter(
                lambda blob: fragments[blob.fragment_identifier].accumulable, blobs
            )

        if self.accumulated.isChecked():
            step = self.accumulation_spinbox.value()
            if step == -1:  # Any step
                return filter(
                    lambda blob: fragments[blob.fragment_identifier].accumulation_step
                    is not None,
                    blobs,
                )

            return filter(
                lambda blob: fragments[blob.fragment_identifier].accumulation_step
                == step,
                blobs,
            )

        if self.not_accumulated.isChecked():
            return filter(
                lambda blob: fragments[blob.fragment_identifier].accumulable
                and fragments[blob.fragment_identifier].accumulation_step is None,
                blobs,
            )
        return ()
