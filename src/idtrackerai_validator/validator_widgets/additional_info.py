from contextlib import suppress

from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget

from idtrackerai import Blob, Fragment
from idtrackerai_GUI_tools import key_event_modifier


class CustomListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setAlternatingRowColors(True)

    def keyPressEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyReleaseEvent(event)


class AdditionalInfo(QWidget):
    fragments: list[Fragment] | None

    def __init__(self) -> None:
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.n_blobs_in_frame = QLabel("")
        self.blob_title = QLabel("Selected blob:")
        self.blob_properties = CustomListWidget()
        self.fragment_title = QLabel("Selected blob's fragment")
        self.fragment_properties = CustomListWidget()
        self.layout().setContentsMargins(0, 0, 0, 8)
        self.layout().addWidget(self.n_blobs_in_frame)
        self.layout().addWidget(self.blob_title)
        self.layout().addWidget(self.blob_properties)
        self.layout().addWidget(self.fragment_title)
        self.layout().addWidget(self.fragment_properties)

    def set_data(self, blob: Blob | None, n_blobs: int):
        self.blob_properties.clear()
        self.fragment_properties.clear()
        # add "All" blobs in frame
        self.n_blobs_in_frame.setText(f"{n_blobs} blobs in frame")
        if blob is None:
            return

        self.blob_properties.addItems(blob.properties)

        if self.fragments is None:
            return

        with suppress(AttributeError):
            self.fragment_properties.addItems(
                self.fragments[blob.fragment_identifier].properties
            )
