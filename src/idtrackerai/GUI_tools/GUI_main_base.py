# Each Qt binding is different, so...
# pyright: reportIncompatibleMethodOverride=false
import logging
from importlib import metadata
from pathlib import Path

from qtpy import API_NAME
from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import QSettings, Qt, QThread, QTimer, QUrl
from qtpy.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
    QDropEvent,
    QGuiApplication,
    QIcon,
)
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStyleFactory,
    QWidget,
)

from idtrackerai.utils.telemetry import (
    check_version,
    get_usage_analytics_state,
    set_usage_analytics_state,
)

from .themes import dark, light


class GUIBase(QMainWindow):

    # in some computers, the tooltip text is white ignoring the palette
    stylesheet: str = (
        "QToolTip { color: black;} QMenu::separator {height: 1px;background: gray; margin-left: 10px; margin-right:10px;}"
    )

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
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "icon.svg")))
        self.settings = QSettings("idtrackerai", "idtrackerai_GUI")
        logging.debug("Saving GUI settings in %s", self.settings.fileName())

        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QHBoxLayout())

        self.documentation_url: str = ""
        """Link to documentation appearing in the menu bar"""

        self.widgets_to_close: list[QWidget] = []
        """Widgets in this list will be called with .close() when closing the app"""

        about_menu = self.menuBar().addMenu("About")
        assert isinstance(about_menu, QMenu)

        doc_action = QAction("Open documentation", self)
        about_menu.addAction(doc_action)
        doc_action.triggered.connect(self.open_docs)

        updates = QAction("Check for updates", self)
        about_menu.addAction(updates)
        updates.triggered.connect(self.check_updates)

        updates = QAction("Report usage analytics", self)
        updates.setCheckable(True)
        updates.setChecked(get_usage_analytics_state())
        about_menu.addAction(updates)
        updates.triggered.connect(set_usage_analytics_state)

        quit = QAction("Quit app", self)
        quit.setShortcut(Qt.Key.Key_Q)
        quit.triggered.connect(self.close)  # type: ignore

        zoom_in = QAction("Zoom in", self)
        zoom_in.setShortcut("Ctrl++")
        zoom_in.triggered.connect(lambda: self.change_font_size(1))  # type: ignore

        zoom_out = QAction("Zoom out", self)
        zoom_out.setShortcut("Ctrl+-")
        zoom_out.triggered.connect(lambda: self.change_font_size(-1))  # type: ignore

        self.themeAction = QAction("Dark theme", self)
        self.themeAction.toggled.connect(self.change_theme)
        self.themeAction.setCheckable(True)
        self.change_theme(False)

        view_menu = self.menuBar().addMenu("View")
        assert view_menu is not None
        view_menu.addAction(quit)
        view_menu.addSeparator()
        view_menu.addAction(zoom_in)
        view_menu.addAction(zoom_out)
        view_menu.addAction(self.themeAction)

        dark_theme = self.settings.value("dark_theme", False, type=bool)
        font_size = self.settings.value("fontsize", self.font().pointSize(), type=int)

        self.themeAction.setChecked(dark_theme)
        font = self.font()
        font.setPointSize(font_size)
        self.setFont(font)
        QApplication.setFont(font)
        self.change_theme(dark_theme)

        self.auto_check_updates = AutoCheckUpdatesThread()
        self.auto_check_updates.out_of_date.connect(
            lambda msg: QMessageBox.about(self, "Check for updates", msg)
        )
        QTimer.singleShot(100, self.auto_check_updates.start)
        self.center_window()
        self.setAcceptDrops(True)

    def change_font_size(self, change: int) -> None:
        font = self.font()
        font.setPointSize(min(max(font.pointSize() + change, 5), 20))
        self.setFont(font)
        QApplication.setFont(font)
        # This has to be here so that the font size change takes place
        self.setStyleSheet(self.stylesheet)

    def check_updates(self):
        out_of_date, message = check_version()
        QMessageBox.about(self, "Check for updates", message)

    def open_docs(self):
        QDesktopServices.openUrl(QUrl(self.documentation_url))

    def center_window(self):
        settings_key = f"{self.__class__.__name__}_window_geometry"
        if self.settings.contains(settings_key):
            self.restoreGeometry(self.settings.value(settings_key))
            return

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

        self.setStyleSheet(self.stylesheet)

    def closeEvent(self, event: QCloseEvent):
        self.settings.setValue("dark_theme", self.themeAction.isChecked())
        self.settings.setValue("fontsize", self.font().pointSize())
        self.settings.setValue(
            f"{self.__class__.__name__}_window_geometry", self.saveGeometry()
        )
        for widget_to_close in self.widgets_to_close:
            widget_to_close.close()
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDropEvent) -> None:
        # accept Drag&Drop if it's about files
        data = event.mimeData()
        if data is not None and data.hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        data = event.mimeData()
        if data is not None and data.hasUrls():
            # send the drop files to self.open_widget
            urls = [url.toLocalFile() for url in data.urls()]
            QTimer.singleShot(0, lambda: self.manageDropedPaths(urls))
            event.accept()

    def manageDropedPaths(self, paths: list[str]) -> None:
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


class AutoCheckUpdatesThread(QThread):
    out_of_date = Signal(str)

    def run(self):
        is_out_of_date, message = check_version()
        if is_out_of_date:
            self.out_of_date.emit(message)
