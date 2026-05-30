"""Segmentation method settings widget for the SegmentationGUI.

Supports three segmentation backends:
- Legacy threshold (default)
- SAM 3 (text-prompted segmentation)
- Detectron2 (instance segmentation with pretrained model)
"""

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class SAM3Widget(QWidget):
    """Widget for selecting and configuring segmentation method.

    Contains:
    - Radio buttons to choose between threshold, SAM 3, and Detectron2
    - SAM 3: text prompt + confidence threshold
    - Detectron2: config path + weights path + confidence threshold + class filter

    Signals
    -------
    methodChanged : str
        Emitted when the segmentation method changes.
        Value is ``"threshold"``, ``"sam3"``, or ``"detectron2"``.
    """

    methodChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # -- Radio buttons for method selection --
        self.radio_threshold = QRadioButton("Threshold (legacy)")
        self.radio_sam3 = QRadioButton("SAM 3")
        self.radio_detectron2 = QRadioButton("Detectron2")
        self.radio_threshold.setChecked(True)

        self.radio_group = QButtonGroup(self)
        self.radio_group.addButton(self.radio_threshold, 0)
        self.radio_group.addButton(self.radio_sam3, 1)
        self.radio_group.addButton(self.radio_detectron2, 2)

        for radio in (self.radio_threshold, self.radio_sam3, self.radio_detectron2):
            radio.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.radio_group.idToggled.connect(self._on_method_toggled)

        radio_row = QHBoxLayout()
        radio_row.addWidget(self.radio_threshold)
        radio_row.addWidget(self.radio_sam3)
        radio_row.addWidget(self.radio_detectron2)

        # -- SAM 3 settings --
        sam3_group = QGroupBox("SAM 3 Settings")
        sam3_layout = QVBoxLayout()

        prompt_label = QLabel("Text prompt:")
        self.text_prompt = QLineEdit()
        self.text_prompt.setPlaceholderText("e.g. zebrafish, ant, mouse")
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(prompt_label)
        prompt_row.addWidget(self.text_prompt)

        sam3_conf_label = QLabel("Confidence:")
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(-5.0, 5.0)
        self.confidence_threshold.setSingleStep(0.1)
        self.confidence_threshold.setValue(0.0)
        self.confidence_threshold.setDecimals(2)
        self.confidence_threshold.setToolTip(
            "Logit threshold for mask acceptance. Higher = stricter."
        )
        sam3_conf_row = QHBoxLayout()
        sam3_conf_row.addWidget(sam3_conf_label)
        sam3_conf_row.addWidget(self.confidence_threshold)

        sam3_layout.addLayout(prompt_row)
        sam3_layout.addLayout(sam3_conf_row)
        sam3_group.setLayout(sam3_layout)
        self.sam3_group = sam3_group

        # -- Detectron2 settings --
        d2_group = QGroupBox("Detectron2 Settings")
        d2_layout = QVBoxLayout()

        config_label = QLabel("Config (required):")
        self.d2_config = QLineEdit()
        self.d2_config.setPlaceholderText(
            "Path to Detectron2 config YAML file"
        )
        self.d2_config_browse = QPushButton("📂")
        self.d2_config_browse.setFixedWidth(32)
        self.d2_config_browse.setToolTip("Browse for config YAML file")
        self.d2_config_browse.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.d2_config_browse.clicked.connect(self._browse_d2_config)
        config_row = QHBoxLayout()
        config_row.addWidget(config_label)
        config_row.addWidget(self.d2_config)
        config_row.addWidget(self.d2_config_browse)

        weights_label = QLabel("Weights (required):")
        self.d2_weights = QLineEdit()
        self.d2_weights.setPlaceholderText(
            "Path to model weights (.pth / .pkl)"
        )
        self.d2_weights_browse = QPushButton("📂")
        self.d2_weights_browse.setFixedWidth(32)
        self.d2_weights_browse.setToolTip("Browse for model weights file")
        self.d2_weights_browse.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.d2_weights_browse.clicked.connect(self._browse_d2_weights)
        weights_row = QHBoxLayout()
        weights_row.addWidget(weights_label)
        weights_row.addWidget(self.d2_weights)
        weights_row.addWidget(self.d2_weights_browse)

        d2_conf_label = QLabel("Confidence:")
        self.d2_confidence = QDoubleSpinBox()
        self.d2_confidence.setRange(0.0, 1.0)
        self.d2_confidence.setSingleStep(0.05)
        self.d2_confidence.setValue(0.5)
        self.d2_confidence.setDecimals(2)
        self.d2_confidence.setToolTip(
            "Score threshold for instance detections. Higher = fewer detections."
        )
        d2_conf_row = QHBoxLayout()
        d2_conf_row.addWidget(d2_conf_label)
        d2_conf_row.addWidget(self.d2_confidence)

        classes_label = QLabel("Classes (required):")
        self.d2_classes = QLineEdit()
        self.d2_classes.setPlaceholderText(
            "Comma-separated class names, e.g. fish, animal"
        )
        self.d2_classes.setToolTip(
            "Class names from your model's training dataset. "
            "E.g. 'fish' or 'zebrafish, medaka'. "
            "For models without named classes, use integer IDs: '0, 2'."
        )
        classes_row = QHBoxLayout()
        classes_row.addWidget(classes_label)
        classes_row.addWidget(self.d2_classes)

        d2_layout.addLayout(config_row)
        d2_layout.addLayout(weights_row)
        d2_layout.addLayout(d2_conf_row)
        d2_layout.addLayout(classes_row)
        d2_group.setLayout(d2_layout)
        self.d2_group = d2_group

        # -- Main layout --
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(radio_row)
        main_layout.addWidget(self.sam3_group)
        main_layout.addWidget(self.d2_group)
        self.setLayout(main_layout)

        # Initial state: threshold mode — hide SAM3 and D2 settings
        self.sam3_group.setVisible(False)
        self.d2_group.setVisible(False)

    def _on_method_toggled(self, button_id: int, checked: bool) -> None:
        """Handle segmentation method radio button toggle."""
        if not checked:
            return

        method = {0: "threshold", 1: "sam3", 2: "detectron2"}.get(
            button_id, "threshold"
        )

        self.sam3_group.setVisible(method == "sam3")
        self.d2_group.setVisible(method == "detectron2")

        self.methodChanged.emit(method)

    def setValue(
        self,
        method: str = "threshold",
        text_prompt: str = "",
        confidence: float = 0.0,
        d2_config: str = "",
        d2_weights: str = "",
        d2_confidence: float = 0.5,
        d2_class_names: list[str] | None = None,
    ) -> None:
        """Set all widget values from session parameters."""
        if method == "sam3":
            self.radio_sam3.setChecked(True)
        elif method == "detectron2":
            self.radio_detectron2.setChecked(True)
        else:
            self.radio_threshold.setChecked(True)

        # SAM 3 values
        self.text_prompt.setText(text_prompt or "")
        self.confidence_threshold.setValue(confidence)

        # Detectron2 values
        self.d2_config.setText(d2_config or "")
        self.d2_weights.setText(d2_weights or "")
        self.d2_confidence.setValue(d2_confidence)
        self.d2_classes.setText(
            ", ".join(d2_class_names) if d2_class_names else ""
        )

    def value(self) -> dict:
        """Return current values as a dict matching session parameter names."""
        if self.radio_sam3.isChecked():
            method = "sam3"
        elif self.radio_detectron2.isChecked():
            method = "detectron2"
        else:
            method = "threshold"

        # Parse class names from comma-separated string
        classes_text = self.d2_classes.text().strip()
        class_names = (
            [c.strip() for c in classes_text.split(",") if c.strip()]
            if classes_text
            else []
        )

        return {
            "segmentation_method": method,
            "sam3_text_prompt": self.text_prompt.text().strip(),
            "sam3_confidence_threshold": self.confidence_threshold.value(),
            "detectron2_config": self.d2_config.text().strip(),
            "detectron2_weights": self.d2_weights.text().strip(),
            "detectron2_confidence_threshold": self.d2_confidence.value(),
            "detectron2_class_names": class_names,
        }

    def isEnabled_sam3(self) -> bool:
        """Return whether SAM 3 is currently enabled."""
        return self.radio_sam3.isChecked()

    def isEnabled_detectron2(self) -> bool:
        """Return whether Detectron2 is currently enabled."""
        return self.radio_detectron2.isChecked()

    def _browse_d2_config(self) -> None:
        """Open a file dialog to select a Detectron2 config YAML file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Detectron2 Config File",
            self.d2_config.text() or "",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.d2_config.setText(path)

    def _browse_d2_weights(self) -> None:
        """Open a file dialog to select Detectron2 model weights."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Detectron2 Weights File",
            self.d2_weights.text() or "",
            "Model weights (*.pth *.pt *.pkl);;All files (*)",
        )
        if path:
            self.d2_weights.setText(path)
