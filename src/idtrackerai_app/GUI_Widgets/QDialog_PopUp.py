from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QDialog,
    QVBoxLayout,
    QStyle,
    QCommonStyle,
)
from PyQt6.QtCore import Qt


class MyMessageBox(QDialog):
    def __init__(self, title=""):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedWidth(300)
        self.setWindowTitle("idTracker.ai")
        self.setLayout(QVBoxLayout())
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignHCenter)
        self.title.setStyleSheet("font-weight: bold")
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignHCenter)
        self.ok = QPushButton("Ok")
        self.ok.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.ok.clicked.connect(super().close)
        self.ok.setIcon(
            QCommonStyle().standardIcon(
                QStyle.StandardPixmap.SP_DialogOkButton
            )
        )
        self.layout().addWidget(self.title)
        self.layout().addWidget(self.text)
        self.layout().addWidget(self.ok, alignment=Qt.AlignRight)

    def exec(self, message=None):
        if message is not None:
            self.text.setText(message)
        return super().exec()
