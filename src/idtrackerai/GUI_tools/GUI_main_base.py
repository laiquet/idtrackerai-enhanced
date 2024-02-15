import json
import logging
from importlib import metadata
from pathlib import Path

from qtpy import API_NAME
from qtpy.QtCore import Qt, QThread, QTimer, QUrl, Signal  # type: ignore
from qtpy.QtGui import QAction, QCloseEvent, QDesktopServices, QGuiApplication, QIcon
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLayout,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.utils import check_version

from .themes import dark, light


class GUIBase(QMainWindow):
    def __init__(self):
        try:
            QT_version = metadata.version(API_NAME)
        except metadata.PackageNotFoundError:
            QT_version = "unknown version"
        logging.info(
            "Initializing %s with %s %s", self.__class__.__name__, API_NAME, QT_version
        )
        if "Fusion" in QStyleFactory.keys():  # noqa SIM118
            QApplication.setStyle("Fusion")
        super().__init__()

        QApplication.setApplicationDisplayName("idtracker.ai")
        QApplication.setApplicationName("idtracker.ai")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "logo_256.png")))

        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QHBoxLayout())

        self.documentation_url: str = ""
        """Link to documentation appearing in the menu bar"""

        self.widgets_to_close: list[QWidget] = []
        """Widgets in this list will be called with .close() when closing the app"""

        about_menu = self.menuBar().addMenu("About")

        doc_action = QAction("Documentation", self)
        about_menu.addAction(doc_action)
        doc_action.triggered.connect(self.open_docs)

        updates = QAction("Check for updates", self)
        about_menu.addAction(updates)
        updates.triggered.connect(self.check_updates)

        fontSizeAction = QAction("Change font size", self)
        fontSizeAction.triggered.connect(lambda: ChangeFontSize(self))  # type: ignore

        quit = QAction("Quit app", self)
        quit.setShortcut(Qt.Key.Key_Q)
        quit.triggered.connect(self.close)  # type: ignore

        self.themeAction = QAction("Dark theme", self)
        self.themeAction.toggled.connect(self.change_theme)
        self.themeAction.setCheckable(True)
        self.change_theme(False)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(quit)
        view_menu.addSeparator()
        view_menu.addAction(fontSizeAction)
        view_menu.addAction(self.themeAction)

        self.json_path = Path(__file__).parent / "QApp_params.json"
        if not self.json_path.is_file():
            self.themeAction.setChecked(False)
        else:
            json_params = json.load(self.json_path.open())
            self.themeAction.setChecked(json_params["dark_theme"])
            font = self.font()
            font.setPointSize(json_params["fontsize"])
            self.setFont(font)
            QApplication.setFont(font)

        # in some computers, the tooltip text is white ignoring the palette
        self.setStyleSheet("QToolTip { color: black;}")

        self.auto_check_updates = AutoCheckUpdatesThread()
        self.auto_check_updates.out_of_date.connect(
            lambda msg: QMessageBox.about(self, "Check for updates", msg)
        )
        QTimer.singleShot(100, self.auto_check_updates.start)
        self.center_window()

    def check_updates(self):
        out_of_date, message = check_version()
        QMessageBox.about(self, "Check for updates", message)

    def open_docs(self):
        QDesktopServices.openUrl(QUrl(self.documentation_url))

    def center_window(self):
        w, h = 1000, 800
        try:
            cp = (
                QGuiApplication.screenAt(self.cursor().pos())
                .availableGeometry()
                .center()
            )
        except AttributeError:
            # in Fedora QGuiApplication.screenAt(self.cursor().pos()) is None
            cp = QGuiApplication.primaryScreen().availableGeometry().center()

        self.setGeometry(cp.x() - w // 2, cp.y() - h // 2, w, h)

    def change_theme(self, dark_theme: bool):
        if dark_theme:
            QApplication.setPalette(dark)
        else:
            QApplication.setPalette(light)

        self.setStyleSheet("QToolTip { color: black;}")

    def closeEvent(self, event: QCloseEvent):
        json.dump(
            {
                "dark_theme": self.themeAction.isChecked(),
                "fontsize": self.font().pointSize(),
            },
            self.json_path.open("w"),
        )
        for widget_to_close in self.widgets_to_close:
            widget_to_close.close()
        super().closeEvent(event)

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


class ChangeFontSize(QDialog):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setFixedSize(300, 50)
        self.setLayout(QVBoxLayout())
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.layout().addWidget(self.slider)
        self.slider.setMinimum(5)
        self.slider.setMaximum(20)
        self.slider.setValue(parent.font().pointSize())
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setValue(self.parent_widget.font().pointSize())
        self.exec()

    def slider_changed(self, value):
        font = self.parent_widget.font()
        font.setPointSize(value)
        self.parent_widget.setFont(font)
        QApplication.setFont(font)

        # This has to be here so that the font size change takes place
        self.parent_widget.setStyleSheet("QToolTip { color: black;}")


class AutoCheckUpdatesThread(QThread):
    out_of_date = Signal(str)

    def run(self):
        is_out_of_date, message = check_version()
        if is_out_of_date:
            self.out_of_date.emit(message)
