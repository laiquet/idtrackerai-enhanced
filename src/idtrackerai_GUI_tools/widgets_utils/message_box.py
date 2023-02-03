from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QVBoxLayout,
)

from .other_utils import WrappedLabel


class MessageBox(QDialog):
    """An implementation of the already existing QMessageBox but with
    adaptative window size"""

    def __init__(self, parent=None, title="", popup_type="info"):
        super().__init__(parent=parent)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMaximumWidth(500)
        self.setWindowTitle("idTracker.ai")

        self.title = WrappedLabel(title)
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.title.setStyleSheet("font-weight: bold")
        self.text = WrappedLabel("")
        self.text.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        ok = QPushButton("Ok")
        ok.clicked.connect(super().accept)
        ok.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_DialogOkButton)
        )

        style = self.style()
        self.infoIcon = style.standardIcon(
            style.StandardPixmap.SP_MessageBoxInformation
        ).pixmap(QSize(64, 64))
        self.warnIcon = style.standardIcon(
            style.StandardPixmap.SP_MessageBoxWarning
        ).pixmap(QSize(64, 64))

        self.big_icon = QLabel()
        right_side = QVBoxLayout()
        right_side.addWidget(self.title)
        right_side.addWidget(self.text)
        right_side.addWidget(ok, alignment=Qt.AlignmentFlag.AlignRight)

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.big_icon)
        layout.addLayout(right_side)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

    def exec(self, warning: bool, title: str, message: str):
        self.big_icon.setPixmap(self.warnIcon if warning else self.infoIcon)
        self.title.setText(title)
        self.text.setText(message)
        return super().exec()
