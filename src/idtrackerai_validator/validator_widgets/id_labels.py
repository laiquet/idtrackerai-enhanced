from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import QFormLayout, QLineEdit, QScrollArea, QWidget


class IdLabels(QScrollArea):
    needToDraw = Signal()
    labels: list[str]

    def __init__(self):
        super().__init__()
        self.form_layout = QFormLayout()
        self.setWidgetResizable(True)
        wid = QWidget()
        wid.setLayout(self.form_layout)
        self.setWidget(wid)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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

    def get_labels(self):
        return self.labels
