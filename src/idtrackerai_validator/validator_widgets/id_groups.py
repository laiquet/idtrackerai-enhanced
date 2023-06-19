from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from idtrackerai_GUI_tools import WrappedLabel

Selected_Color = QColor(255, 0, 0)
Unselected_Color = QColor(255, 255, 255)
Selected_Color_alpha = QColor(255, 0, 0, 75)
Unselected_Color_alpha = QColor(255, 255, 255, 75)


class IdGroups(QWidget):
    needToDraw = Signal()
    unsaved_changes = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        first_row = QHBoxLayout()
        first_row.addWidget(QLabel("Identity groups"))
        self.add_btn = QToolButton()
        self.add_btn.setText("Add")
        first_row.addWidget(self.add_btn)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.main_layout.addLayout(first_row)
        self.add_btn.clicked.connect(self.add_clicked)
        self.editing_name: str = ""
        self.view: set[str] = set()
        self.id_groups: dict[str, tuple[QWidget, set[int]]] = {}

    def generate_row(self, name: str, group: set[int]):
        label = WrappedLabel(f"{name}: {', '.join(map(str,group))}")
        label.setObjectName("label")
        label.setWordWrap(True)
        view_btn = QToolButton()
        view_btn.setText("View")
        view_btn.setObjectName("view")
        view_btn.setCheckable(True)
        view_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        view_btn.toggled.connect(
            lambda c: self.view_btn_clicked(view_btn, label.text().split(":")[0], c)
        )
        edit_btn = QToolButton()
        edit_btn.setText("Edit")
        edit_btn.setObjectName("edit")
        edit_btn.setCheckable(True)
        edit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        edit_btn.toggled.connect(
            lambda c: self.edit_btn_clicked(edit_btn, label.text().split(":")[0], c)
        )
        remove_btn = QToolButton()
        remove_btn.setText("-")
        remove_btn.setObjectName("remove")
        remove_btn.setToolTip("Remove")
        remove_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove_btn.clicked.connect(
            lambda: self.remove_btn_clicked(label.text().split(":")[0])
        )

        row = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(view_btn)
        layout.addWidget(edit_btn)
        layout.addWidget(remove_btn)
        row.setLayout(layout)
        self.main_layout.addWidget(row)
        return row

    def uncheck_btns(self, exception: QToolButton | None):
        for btns, _group in self.id_groups.values():
            for widget in btns.findChildren(QToolButton):
                assert isinstance(widget, QToolButton)
                if widget != exception and widget.isChecked():
                    widget.setChecked(False)

    def view_btn_clicked(self, btn: QToolButton, name: str, checked: bool):
        if checked:
            self.uncheck_btns(btn)
            self.view.add(name)
        else:
            self.view.remove(name)
        self.needToDraw.emit()

    def edit_btn_clicked(self, btn: QToolButton, name: str, checked: bool):
        if checked:
            self.uncheck_btns(btn)
        self.editing_name = name if checked else ""
        self.needToDraw.emit()

    def remove_btn_clicked(self, name: str):
        self.uncheck_btns(None)
        row = self.id_groups.pop(name)[0]
        self.main_layout.removeWidget(row)

    def load_groups(self, identities_groups: dict):
        while self.id_groups:
            row = self.id_groups.popitem()[1][0]
            self.main_layout.removeWidget(row)

        self.id_groups = {
            key: (self.generate_row(key, set(value)), set(value))
            for key, value in identities_groups.items()
        }

    def selected_id(self, identity: int | None):
        if not self.editing_name or identity in (-1, None):
            return

        row, group = self.id_groups[self.editing_name]
        if identity in group:
            group.remove(identity)
        else:
            group.add(identity)
        label = row.findChild(WrappedLabel, "label")
        assert isinstance(label, WrappedLabel)
        label.setText(f"{self.editing_name}: {', '.join(map(str,group))}")
        self.unsaved_changes.emit()

    def add_clicked(self):
        name, ok = QInputDialog.getText(
            self, "idtracker.ai", "Enter identity group name:"
        )
        name = name.strip()

        if not ok or not name:
            return

        if name in self.id_groups:
            edit_btn = self.id_groups[name][0].findChild(QToolButton, "edit")
            assert isinstance(edit_btn, QToolButton)
            edit_btn.setChecked(True)
            return

        btns, group = self.generate_row(name, set()), set()
        self.id_groups[name] = btns, group
        edit = btns.findChild(QToolButton, "edit")
        assert isinstance(edit, QToolButton)
        edit.setChecked(True)

    def is_active(self) -> bool:
        return (
            self.isVisible()
            and self.isEnabled()
            and (bool(self.editing_name) or bool(self.view))
        )

    def get_groups(self):
        return {key: value[1] for key, value in self.id_groups.items()}

    def get_cmaps(self, n_animals: int):
        names = [self.editing_name] if self.editing_name else self.view
        cmap = [Unselected_Color] * (n_animals + 1)
        cmap_alpha = [Unselected_Color_alpha] * (n_animals + 1)

        for name in names:
            group = self.id_groups[name][1]
            for identity in group:
                cmap[identity] = Selected_Color
                cmap_alpha[identity] = Selected_Color_alpha

        return cmap, cmap_alpha

    def uncheck_edit_buttons(self):
        for edit_btn in self.findChildren(QToolButton, "edit"):
            assert isinstance(edit_btn, QToolButton)
            edit_btn.setChecked(False)
