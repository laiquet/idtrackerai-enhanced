import json
import logging
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction, QGuiApplication, QIcon, QKeyEvent
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLayout, QMainWindow, QWidget

from . import ChangeFontSize, custom, light


class GUIBase(QMainWindow):
    def __init__(self):
        logging.debug(f"Initializing {self.__class__.__name__}")
        super().__init__()

        self.setWindowTitle("idTracker.ai")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "logo_256.png")))

        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QHBoxLayout())

        view_menu = self.menuBar().addMenu("View")

        fontSizeAction = QAction("Change font size", self)
        view_menu.addAction(fontSizeAction)
        fontSizeAction.triggered.connect(lambda: ChangeFontSize(self))

        self.themeAction = QAction("Dark theme", self)
        self.themeAction.toggled.connect(self.change_theme)
        self.themeAction.setCheckable(True)
        self.change_theme(False)
        view_menu.addAction(self.themeAction)

        self.json_path = Path(__file__).parent / "QApp_params.json"
        if not self.json_path.is_file():
            self.themeAction.setChecked(False)
            self.font().pointSize()
        else:
            json_params = json.load(self.json_path.open())
            self.themeAction.setChecked(json_params["dark_theme"])
            font = self.font()
            font.setPointSize(json_params["fontsize"])
            self.setFont(font)

    def center_window(self):
        w, h = 1000, 800
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        self.setGeometry(cp.x() - w // 2, cp.y() - h // 2, w, h)

    def change_theme(self, dark: bool):
        if dark:
            QApplication.setPalette(custom)
        else:
            QApplication.setPalette(light)

    def keyPressEvent(self, event: QKeyEvent):
        if hasattr(event, "isAutoRepeat") and event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key.Key_Q:
            self.close()
        self.processed_keyPressEvent(key)

    def closeEvent(self, event: QEvent):
        json.dump(
            dict(
                dark_theme=self.themeAction.isChecked(),
                fontsize=self.font().pointSize(),
            ),
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
