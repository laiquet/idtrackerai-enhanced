from collections.abc import Iterable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QRadioButton, QScrollArea, QVBoxLayout, QWidget

from idtrackerai import Blob, Fragment


class MarkBlobs(QScrollArea):
    needToDraw = pyqtSignal()

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
        self.accumulated = QRadioButton("Accumulated")
        self.forces_to_crossing = QRadioButton("Forced to be crossing")

        self.individual.toggled.connect(self.needToDraw.emit)
        self.used_for_training.toggled.connect(self.needToDraw.emit)
        self.used_for_training_crossings.toggled.connect(self.needToDraw.emit)
        self.accumulable.toggled.connect(self.needToDraw.emit)
        self.accumulated.toggled.connect(self.needToDraw.emit)
        self.not_accumulated.toggled.connect(self.needToDraw.emit)
        self.forces_to_crossing.toggled.connect(self.needToDraw.emit)

        main_layout.addWidget(self.individual)
        main_layout.addWidget(self.used_for_training)
        main_layout.addWidget(self.used_for_training_crossings)
        main_layout.addWidget(self.accumulable)
        main_layout.addWidget(self.accumulated)
        main_layout.addWidget(self.not_accumulated)
        main_layout.addWidget(self.forces_to_crossing)

    def __call__(
        self, blobs: list[Blob], fragments: list[Fragment] | None
    ) -> Iterable[Blob]:
        if not self.isVisible() or not self.isEnabled():
            return ()

        if self.individual.isChecked():
            return filter(lambda blob: blob.is_an_individual, blobs)

        if self.crossing.isChecked():
            return filter(lambda blob: blob.is_a_crossing, blobs)

        if self.used_for_training.isChecked():
            return filter(lambda blob: blob.used_for_training, blobs)

        if self.used_for_training_crossings.isChecked():
            return filter(lambda blob: blob.used_for_training_crossings, blobs)

        if fragments is None:
            return ()

        if self.accumulable.isChecked():
            return filter(
                lambda blob: fragments[blob.fragment_identifier].accumulable, blobs
            )

        if self.accumulated.isChecked():
            return filter(lambda blob: blob.accumulation_step is not None, blobs)

        if self.not_accumulated.isChecked():
            return filter(
                lambda blob: fragments[blob.fragment_identifier].accumulable
                and blob.accumulation_step is None,
                blobs,
            )
        return ()
