import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from idtrackerai_GUI_tools import LabeledSlider, key_event_modifier


class CustomTableWidget(QTableWidget):
    def keyPressEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, e: QKeyEvent):
        event = key_event_modifier(e)
        if event is not None:
            super().keyReleaseEvent(event)


class CustomTableWidgetItem(QTableWidgetItem):
    def __init__(self, value: str | int):
        super().__init__("" if value == -1 else str(value))
        self.setData(Qt.ItemDataRole.UserRole, value)
        self.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        return self.data(Qt.ItemDataRole.UserRole) < other.data(
            Qt.ItemDataRole.UserRole
        )


class ErrorsExplorer(QWidget):
    go_to_error = pyqtSignal(str, int, int, object, int)
    # kind, start, end, where, id

    def __init__(self):
        super().__init__()
        self.table = CustomTableWidget(1, 4)
        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        horizontalHeader.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        horizontalHeader.setMinimumSectionSize(10)
        verticalHeader = self.table.verticalHeader()
        verticalHeader.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        verticalHeader.setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(["Type", "Id", "Start", "Length"])
        self.table.setSortingEnabled(True)
        self.table.currentCellChanged.connect(self.cell_clicked)
        self.table.cellDoubleClicked.connect(self.cell_clicked)

        long_jumps_row = QHBoxLayout()
        long_jumps_row.addWidget(QLabel("Jumps threshold"))
        self.long_jumps_th = LabeledSlider(self, 3, 10)
        long_jumps_row.addWidget(self.long_jumps_th)
        self.long_jumps_th.setValue(5)
        self.long_jumps_th.valueChanged.connect(self.update_list_of_errors)

        layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(layout)
        errors_header = QHBoxLayout()
        self.update_btn = QToolButton()
        self.update_btn.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
        )
        self.update_btn.clicked.connect(self.update_list_of_errors)
        self.left_label = QLabel()
        errors_header.addWidget(self.left_label)
        errors_header.addWidget(self.update_btn)
        layout.addLayout(errors_header)
        layout.addWidget(self.table)
        layout.addLayout(long_jumps_row)
        self.setLayout(layout)

    def cell_clicked(self, row: int, col: int):
        if row < 0 or col < 0:
            return

        kind = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        id = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        start = self.table.item(row, 2).data(Qt.ItemDataRole.UserRole)
        length = self.table.item(row, 3).data(Qt.ItemDataRole.UserRole)

        where = None
        if kind in ("Jump", "Miss id"):
            if start > 0:
                where = self.trajectories[start - 1 : start + length + 1, id - 1]
            else:
                where = self.trajectories[start : start + length + 1, id - 1]
        self.go_to_error.emit(kind, start, length, where, id)

    def set_references(
        self, traj: np.ndarray, all_identified: np.ndarray, duplicated: np.ndarray
    ):
        self.trajectories = traj
        self.all_identified = all_identified
        self.duplicated = duplicated
        self.update_btn.click()

    def getErrors(self) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
        # TODO Add more errors (super-crossings)
        return {
            "Miss id": get_list_of_Trues_for_id(np.isnan(self.trajectories[..., 0])),
            "No Id": [(-1,) + get_list_of_Trues(self.all_identified)],
            "Dupl": get_list_of_Trues_for_id(self.duplicated),
            "Jump": get_impossible_jumps(self.trajectories, self.long_jumps_th.value()),
        }

    def update_list_of_errors(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for error_kind, errors_for_id in self.getErrors().items():
            for id, starts, lengths in errors_for_id:
                for start, length in zip(starts, lengths):
                    self.table.insertRow(0)
                    self.table.setItem(0, 0, CustomTableWidgetItem(error_kind))
                    self.table.setItem(0, 1, CustomTableWidgetItem(id))
                    self.table.setItem(0, 2, CustomTableWidgetItem(start))
                    self.table.setItem(0, 3, CustomTableWidgetItem(length))
        self.left_label.setText(f"List of errors ({self.table.rowCount()} errors)")
        self.table.setSortingEnabled(True)


def get_list_of_Trues(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the start and the length of every True cluster in an array

    Input: [0,1,0,1,1,0]
    Output [1,3], [1,2] (one cluster at 1 of length 1 and another at 3 of length 2)
    """
    where = arr.nonzero()[0]
    is_edge = np.diff(where, prepend=-np.inf, append=np.inf) > 1
    starts = where[is_edge[:-1]]
    ends = where[is_edge[1:]]
    return starts, ends - starts + 1


def get_list_of_Trues_for_id(
    data: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    return [
        (fish_id + 1,) + get_list_of_Trues(data[:, fish_id])
        for fish_id in range(data.shape[1])
    ]


def get_impossible_jumps(traj: np.ndarray, sigma: float = 4.0):
    speed = np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=-1))
    mean, std = np.nanmean(speed), np.nanstd(speed)
    out = get_list_of_Trues_for_id(speed > (mean + sigma * std))
    for id, start, length in out:
        start += 1
    return out
