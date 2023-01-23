import logging

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QAction, QGuiApplication, QKeyEvent, QIcon
from PyQt6.QtWidgets import QHBoxLayout, QLayout, QMainWindow, QWidget, QApplication
from pathlib import Path
from . import ChangeFontSize, custom, light
import json


class GUIBase(QMainWindow):
    def __init__(self):
        logging.debug(f"Initializing {self.__class__.__name__}")
        super().__init__()

        self.setWindowTitle("idTracker.ai")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "logo_256.png")))

        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QHBoxLayout())

        fontSizeAction = QAction("Change font size", self)
        self.menuBar().addAction(fontSizeAction)
        fontSizeAction.triggered.connect(lambda: ChangeFontSize(self))

        themeAction = QAction("Change theme", self)
        self.menuBar().addAction(themeAction)

        self.json_path = Path(__file__).parent / "QApp_params.json"
        if not self.json_path.is_file():
            self.dark_theme = True
            self.font().pointSize()
        else:
            json_params = json.load(self.json_path.open())
            self.dark_theme = not json_params["dark_theme"]
            font = self.font()
            font.setPointSize(json_params["fontsize"])
            self.setFont(font)
        self.change_theme()

        themeAction.triggered.connect(self.change_theme)

    def center_window(self):
        w, h = 1000, 800
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        self.setGeometry(cp.x() - w // 2, cp.y() - h // 2, w, h)

    def change_theme(self):
        if self.dark_theme:
            QApplication.setPalette(light)
        else:
            QApplication.setPalette(custom)
        self.dark_theme = not self.dark_theme

    def keyPressEvent(self, event: QKeyEvent):
        if hasattr(event, "isAutoRepeat") and event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key.Key_Q:
            self.close()
        self.processed_keyPressEvent(key)

    def closeEvent(self, event: QEvent):
        json.dump(
            dict(dark_theme=self.dark_theme, fontsize=self.font().pointSize()),
            self.json_path.open("w"),
        )
        event.accept()

    def processed_keyPressEvent(self, key: int):
        raise NotImplementedError

    def keyReleaseEvent(self, event: QKeyEvent):
        if hasattr(event, "isAutoRepeat"):
            if event.isAutoRepeat():
                return
        self.processed_keyReleaseEvent(event.key())

    def processed_keyReleaseEvent(self, key: int):
        raise NotImplementedError

    def clearFocus(self):
        focused_widged = self.focusWidget()
        if focused_widged:
            focused_widged.clearFocus()

    def mousePressEvent(self, event):
        self.clearFocus()
        super().mousePressEvent(event)

    @staticmethod
    def get_list_of_widgets(layout: QLayout) -> list[QWidget]:
        widgets = []
        layouts = [layout]
        while layouts:
            element = layouts.pop()
            if hasattr(element.widget(), "setEnabled"):
                widgets.append(element.widget())
            else:
                layouts += [element.itemAt(i) for i in range(element.count())]
        return widgets
