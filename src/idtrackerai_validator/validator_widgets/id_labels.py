from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFormLayout, QLineEdit, QScrollArea, QWidget


class IdLabels(QScrollArea):
    needToDraw = pyqtSignal()
    labels: list[str]

    def __init__(self):
        super().__init__()
        self.form_layout = QFormLayout()
        self.setWidgetResizable(True)
        wid = QWidget()
        wid.setLayout(self.form_layout)
        self.setWidget(wid)

        # To disable horizontal scrolling
        self.setMinimumWidth(
            self.widget().minimumSizeHint().width()
            + self.verticalScrollBar().width()
            + 2  # it works with this extra padding
        )

    def load_labels(self, labels: list[str]):
        for _ in range(self.form_layout.rowCount()):
            self.form_layout.removeRow(0)

        for indx, label in enumerate(labels, 1):
            edit = QLineEdit()
            edit.setText(label)
            edit.setPlaceholderText(str(indx))
            edit.setObjectName(str(indx))
            edit.textChanged.connect(self.new_label)
            edit.editingFinished.connect(self.validate_label)
            self.form_layout.addRow(f"{indx}:", edit)

        self.labels = [""] + labels

    def validate_label(self):
        sender = self.sender()
        assert isinstance(sender, QLineEdit)
        text = sender.text()
        if not text:
            sender.setText(sender.placeholderText())
        else:
            sender.setText(text.strip())

    def new_label(self, new_label=""):
        self.labels[int(self.sender().objectName())] = new_label
        self.needToDraw.emit()

    def change_label(self, id: int):
        """This could be used to change the selected
        id label by clicking on the canvas"""

    def get_labels(self):
        return self.labels
