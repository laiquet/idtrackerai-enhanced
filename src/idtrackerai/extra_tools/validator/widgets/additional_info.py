from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget
from superqt import QToggleSwitch

from idtrackerai import Blob, Fragment
from idtrackerai.GUI_tools import GUIBase, key_event_modifier


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

    def __init__(self, parent: GUIBase) -> None:
        super().__init__()
        self.settings = parent.settings
        self.settings_key = f"{parent.__class__.__name__}/show_metadata_checked"
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.metadata_visibility = QToggleSwitch("Show metadata")
        self.metadata_visibility.toggled.connect(self.toggle_groups_visibility)

        self.blob_title = QLabel("Selected blob:")
        self.blob_properties = CustomListWidget()
        self.fragment_title = QLabel("Selected blob's fragment")
        self.fragment_properties = CustomListWidget()
        layout.setContentsMargins(0, 0, 0, 8)
        layout.addWidget(self.metadata_visibility)
        layout.addWidget(self.blob_title)
        layout.addWidget(self.blob_properties)
        layout.addWidget(self.fragment_title)
        layout.addWidget(self.fragment_properties)

        self.metadata_visibility.setChecked(
            self.settings.value(self.settings_key, False, type=bool)
        )
        self.toggle_groups_visibility(self.metadata_visibility.isChecked())

    def toggle_groups_visibility(self, checked: bool):
        self.settings.setValue(self.settings_key, checked)
        self.blob_title.setVisible(checked)
        self.blob_properties.setVisible(checked)
        self.fragment_title.setVisible(checked)
        self.fragment_properties.setVisible(checked)

    def set_data(self, blob: Blob | None):
        self.blob_properties.clear()
        self.fragment_properties.clear()
        if blob is None:
            return

        try:
            self.blob_properties.addItems(blob.summary)
        except AttributeError:
            self.blob_properties.addItem("Corrupted Blob")

        if self.fragments is None:
            return

        try:
            self.fragment_properties.addItems(
                self.fragments[blob.fragment_identifier].summary
            )
        except AttributeError:
            self.fragment_properties.addItem("Corrupted Fragment")
