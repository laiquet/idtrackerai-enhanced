from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QStyle,
    QCommonStyle,
    QLayout,
)
from PyQt6.QtCore import Qt, QSize
from .other_utils import WrappedLabel


class MessageBox(QDialog):
    """An implementation of the already existing QMessageBox but with
    adaptative window size"""

    def __init__(self, parent=None, title="", popup_type="info"):
        super().__init__(parent=parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMaximumWidth(500)
        self.setWindowTitle("idTracker.ai")

        title = WrappedLabel(title)
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet("font-weight: bold")
        self.text = WrappedLabel("")
        self.text.setAlignment(Qt.AlignHCenter)
        ok = QPushButton("Ok")
        ok.clicked.connect(super().accept)
        ok.setIcon(
            QCommonStyle().standardIcon(
                QStyle.StandardPixmap.SP_DialogOkButton
            )
        )

        if popup_type == "info":
            icon = QStyle.StandardPixmap.SP_MessageBoxInformation
        elif popup_type == "warning":
            icon = QStyle.StandardPixmap.SP_MessageBoxWarning

        big_icon = QLabel()
        big_icon.setPixmap(
            QCommonStyle().standardIcon(icon).pixmap(QSize(64, 64))
        )

        right_side = QVBoxLayout()
        right_side.addWidget(title)
        right_side.addWidget(self.text)
        right_side.addWidget(ok, alignment=Qt.AlignRight)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(big_icon)
        self.layout().addLayout(right_side)
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

    def exec(self, message):
        self.text.setText(message)
        return super().exec()
