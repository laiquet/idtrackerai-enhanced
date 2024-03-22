import numpy as np
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QKeyEvent, QPainterPath, QPalette, QPolygonF, QResizeEvent
from qtpy.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QSizePolicy, QStyle

from idtrackerai.utils import get_vertices_from_label


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
        self.icon.setPixmap(
            self.style()
            .standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            .pixmap(70, 70)
        )
        self.text.setText(f"<strong><center>{title}</strong></center><br><br>{text}")
        self.exec()

    def info(self, title: str, text) -> None:
        self.icon.setPixmap(
            self.style()
            .standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
            .pixmap(70, 70)
        )
        self.text.setText(f"<strong><center>{title}</strong></center><br><br>{text}")
        self.exec()

    def keyPressEvent(self, *args, **kwargs) -> None:
        self.close()


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
    list_of_ROIs: list[str] | str | None,
    resolution_reduction: float,
    width: int,
    height: int,
) -> QPainterPath:
    path = QPainterPath()

    if list_of_ROIs is None:
        return path

    if isinstance(list_of_ROIs, str):
        list_of_ROIs = [list_of_ROIs]

    path.addRect(
        -0.5, -0.5, width * resolution_reduction, height * resolution_reduction
    )

    for line in list_of_ROIs:
        path_i = get_path_from_points(
            get_vertices_from_label(line), resolution_reduction
        )

        if line[0] == "+":
            path -= path_i
        elif line[0] == "-":
            path += path_i
        else:
            raise TypeError
    return path


def get_path_from_points(points: np.ndarray, res_reduct: float = 1) -> QPainterPath:
    points = points * res_reduct + 0.5

    path = QPainterPath()
    if points.ndim == 2:
        # some polygons are made from a single point, 1 dimension
        path.addPolygon(QPolygonF(QPointF(*point) for point in points))
    return path.simplified()


class WrappedLabel(QLabel):
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
