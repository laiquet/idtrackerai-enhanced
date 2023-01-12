import logging

from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QAction, QGuiApplication, QKeyEvent
from PyQt6.QtWidgets import QHBoxLayout, QLayout, QMainWindow, QWidget

from . import ChangeFontSize, apply_style


class GUIBase(QMainWindow):
    def __init__(self):
        logging.debug(f"Initializing {self.__class__.__name__}")
        super().__init__()

        self.setWindowTitle("idTracker.ai")

        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QHBoxLayout())

        fontSizeAction = QAction("Change font size", self)
        self.menuBar().addAction(fontSizeAction)
        fontSizeAction.triggered.connect(lambda: ChangeFontSize(self))

        themeAction = QAction("Change theme", self)
        self.menuBar().addAction(themeAction)

        self.dark_theme = False
        themeAction.triggered.connect(self.change_theme)

    def center_window(self):
        w, h = 1000, 800
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        self.setGeometry(cp.x() - w // 2, cp.y() - h // 2, w, h)

    def change_theme(self):
        if self.dark_theme:
            apply_style(self, "light")
        else:
            apply_style(self, "custom")
        self.dark_theme = not self.dark_theme

    def keyPressEvent(self, event: QKeyEvent):
        if hasattr(event, "isAutoRepeat") and event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key.Key_Q:
            QCoreApplication.quit()
        self.processed_keyPressEvent(key)

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
