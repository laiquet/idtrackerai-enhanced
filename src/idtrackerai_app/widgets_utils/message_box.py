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


class MessageBox(QDialog):
    """An implementation of the already existing QMessageBox but with
    adaptative window size"""

    message_width = 300

    def __init__(self, parent=None, title="", popup_type="info"):
        super().__init__(parent=parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("idTracker.ai")

        title = QLabel(title)
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet("font-weight: bold")
        title.setWordWrap(True)
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignHCenter)
        self.ok = QPushButton("Ok")
        self.ok.clicked.connect(super().close)
        self.ok.setIcon(
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
        right_side.addWidget(self.ok, alignment=Qt.AlignRight)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(big_icon)
        self.layout().addLayout(right_side)
        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self.text.setFixedWidth(self.message_width)

    def exec(self, message):
        self.text.setText(message)
        self.text.setFixedHeight(self.text.heightForWidth(self.text.width()))
        return super().exec()
