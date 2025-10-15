# Each Qt binding is different, so...
# pyright: reportIncompatibleMethodOverride=false
from pathlib import Path
from typing import Literal

import numpy as np
from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import QEvent, QPointF, QSettings, Qt
from qtpy.QtGui import (
    QIcon,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPolygonF,
    QResizeEvent,
)
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QStyle,
    QToolButton,
    QWidget,
)

from idtrackerai import Session
from idtrackerai.utils import IdtrackeraiError, get_vertices_from_label

ADD_ICON = QIcon.fromTheme("list-add")
REMOVE_ICON = QIcon.fromTheme("edit-clear")

_theme_icons = {
    "cancel": QIcon.fromTheme("dialog-cancel"),
    "ok": QIcon.fromTheme("dialog-ok"),
    "add": QIcon.fromTheme("list-add"),
    "remove": QIcon.fromTheme("edit-clear"),
    "run": QIcon.fromTheme("system-run"),
    "refresh": QIcon.fromTheme("view-refresh"),
    "undo": QIcon.fromTheme("edit-undo"),
}
_style_icons = {
    "cancel": QStyle.StandardPixmap.SP_DialogCancelButton,
    "ok": QStyle.StandardPixmap.SP_DialogOkButton,
    "add": QStyle.StandardPixmap.SP_FileDialogNewFolder,
    "remove": QStyle.StandardPixmap.SP_TrashIcon,
    "run": QStyle.StandardPixmap.SP_ArrowRight,
    "refresh": QStyle.StandardPixmap.SP_BrowserReload,
    "undo": QStyle.StandardPixmap.SP_ArrowBack,
}


def get_icon(
    icon_name: Literal["cancel", "ok", "add", "remove", "run", "refresh", "undo"],
) -> QIcon:
    theme_icon = _theme_icons[icon_name]
    style = QApplication.style()
    if theme_icon.isNull() and style is not None:
        return style.standardIcon(_style_icons[icon_name])
    return theme_icon


class AddBtn(QToolButton):
    """A button to add items"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if ADD_ICON.isNull():
            self.setText("Add")
        else:
            self.setIcon(ADD_ICON)
        self.setToolTip("Add")


class RemoveBtn(QToolButton):
    """A button to remove items"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if REMOVE_ICON.isNull():
            self.setText("Remove")
        else:
            self.setIcon(REMOVE_ICON)
        self.setToolTip("Remove")


class LightPopUp(QDialog):
    """A light version of QMessageBox.warning()"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setLayout(QHBoxLayout())
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.icon = QLabel()

        self.icon.setFixedSize(100, 100)
        self.setFixedWidth(500)

        self.layout().addWidget(self.icon)
        self.layout().addWidget(self.text)

    def warning(self, title: str, text) -> None:
        self.icon.setPixmap(QIcon.fromTheme("dialog-warning").pixmap(70, 70))
        self.text.setText(f"<strong><center>{title}</strong></center><br><br>{text}")
        self.exec()

    def info(self, title: str, text) -> None:
        self.icon.setPixmap(QIcon.fromTheme("dialog-information").pixmap(70, 70))
        self.text.setText(f"<strong><center>{title}</strong></center><br><br>{text}")
        self.exec()

    def keyPressEvent(self, *args, **kwargs) -> None:
        self.close()


def open_session(
    app: QWidget, settings: QSettings, session_path: str | Path | None
) -> Session | None:

    if not session_path or isinstance(session_path, bool):
        file_dialog = QFileDialog()
        settings_key = f"{app.__class__.__name__}_filedialog_state"
        state = settings.value(settings_key, b"")
        if state:
            file_dialog.restoreState(state)
        session_path = file_dialog.getExistingDirectory(
            app, "Open session directory", options=QFileDialog.Option.ShowDirsOnly
        )
        settings.setValue(settings_key, file_dialog.saveState())
        if not session_path:
            return

    user_folder = None
    while True:
        try:
            session = Session.load(
                session_path, user_folder, allow_not_found_video_files=False
            )
        except (FileNotFoundError, ValueError) as exc:
            # Provably the session JSON file was not found or was not JSON readable
            QMessageBox.warning(app, "Loading session error", str(exc))
            return
        except IdtrackeraiError as exc:
            # The video files were not found
            response = QMessageBox.warning(
                app,
                "Video files not found",
                f"{exc}\n\nPlease, open the directory containing the video paths to proceed.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Open,
            )
            if response != QMessageBox.StandardButton.Open:
                return

            file_dialog = QFileDialog()
            settings_key = f"{app.__class__.__name__}_filedialog_state"
            state = settings.value(settings_key, b"")
            if state:
                file_dialog.restoreState(state)
            user_folder = file_dialog.getExistingDirectory(
                app,
                "Select the folder containing the video files",
                options=QFileDialog.Option.ShowDirsOnly,
            )
            settings.setValue(settings_key, file_dialog.saveState())
        else:
            return session


class QHLine(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setContentsMargins(10, 0, 10, 0)
        self.setEnabled(False)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.EnabledChange:
            self.setEnabled(False)


def build_ROI_patches_from_list(
    list_of_ROIs: list[str] | str | None, width: int, height: int
) -> QPainterPath:
    path = QPainterPath()

    if list_of_ROIs is None:
        return path

    if isinstance(list_of_ROIs, str):
        list_of_ROIs = [list_of_ROIs]

    path.addRect(-0.5, -0.5, width, height)

    for line in list_of_ROIs:
        path_i = get_path_from_points(get_vertices_from_label(line))

        if line[0] == "+":
            path -= path_i
        elif line[0] == "-":
            path += path_i
        else:
            raise TypeError
    return path


def get_path_from_points(points: np.ndarray) -> QPainterPath:
    path = QPainterPath()
    if points.ndim == 2:
        # some polygons are made from a single point, 1 dimension
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    return path.simplified()


class WrappedLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            and not self.hasSelectedText()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def __init__(
        self,
        text: str = "",
        framed: bool = False,
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
    ) -> None:
        super().__init__(text)
        if framed:
            self.setBackgroundRole(QPalette.ColorRole.Base)
            self.setAutoFillBackground(True)
            self.setContentsMargins(5, 3, 5, 3)
        self.setAlignment(align)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def set_size(self) -> None:
        self.setMinimumHeight(0)
        self.setMinimumHeight(max(self.heightForWidth(self.width()), 1))

    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.set_size()
        super().resizeEvent(a0)

    def setText(self, text: str) -> None:
        # Add Zero-width space in backslashes for proper word wrapping
        super().setText(text.replace("\\", "\\\u200b"))
        self.set_size()

    def text(self):
        return super().text().replace("\u200b", "")


_ignored_keys = {
    Qt.Key.Key_D,
    Qt.Key.Key_A,
    Qt.Key.Key_Left,
    Qt.Key.Key_Right,
    Qt.Key.Key_0,
    Qt.Key.Key_1,
    Qt.Key.Key_2,
    Qt.Key.Key_3,
    Qt.Key.Key_4,
    Qt.Key.Key_5,
    Qt.Key.Key_6,
    Qt.Key.Key_7,
    Qt.Key.Key_8,
    Qt.Key.Key_9,
}


def key_event_modifier(event: QKeyEvent) -> QKeyEvent | None:
    event.key().numerator
    if event.key() == Qt.Key.Key_W:
        return QKeyEvent(event.type(), Qt.Key.Key_Up, event.modifiers())
    if event.key() == Qt.Key.Key_S:
        return QKeyEvent(event.type(), Qt.Key.Key_Down, event.modifiers())
    if event.key() in _ignored_keys:
        # These keys would be accepted by QTableWidget
        # but we want them to control the VideoPlayer
        event.ignore()
        return None
    return event


class TransparentDisabledOverlay(QWidget):
    """Transparent widget to add to another one which will display self.text when the other is disabled"""

    def __init__(self, text: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.text = text
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.parent_widget = parent
        parent.installEventFilter(self)

    def eventFilter(self, widget, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parent_widget.rect())
        return False

    def paintEvent(self, a0) -> None:
        if self.isEnabled():
            return
        painter = QPainter(self)
        painter.setPen(
            self.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text)
        )
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text)
