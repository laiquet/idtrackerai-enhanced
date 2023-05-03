from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget

from idtrackerai import Blob, ListOfFragments
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
    list_of_fragments: ListOfFragments | None

    def __init__(self) -> None:
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.blob_title = QLabel("Selected blob:")
        self.blob_properties = CustomListWidget()
        # self.fragment_title = QLabel("Selected blob's fragment")
        # self.fragment_properties = CustomListWidget()
        self.layout().setContentsMargins(0, 0, 0, 8)
        self.layout().addWidget(self.blob_title)
        self.layout().addWidget(self.blob_properties)
        # self.layout().addWidget(self.fragment_title)
        # self.layout().addWidget(self.fragment_properties)

    def set_data(self, blob: Blob | None):
        self.blob_properties.clear()
        # self.fragment_properties.clear()
        if blob is None:
            return

        self.blob_properties.addItems(blob.properties)

        # if self.list_of_fragments is None:
        #     return

        # self.fragment_properties.addItems(
        #     self.list_of_fragments.fragments[blob.fragment_identifier].properties
        # )
