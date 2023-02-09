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
        self.table = CustomTableWidget(1, 5)
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
        self.table.setHorizontalHeaderLabels(["Type", "Id", "Start", "End", "Length"])
        self.table.setSortingEnabled(True)
        self.table.currentCellChanged.connect(self.cell_clicked)
        self.table.cellClicked.connect(self.cell_clicked)

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
        kind, id, start, end, length = [
            self.table.item(row, col).data(Qt.ItemDataRole.UserRole) for col in range(5)
        ]
        where = None
        if kind in ("Jump", "Miss id"):
            if start > 0:
                where = self.trajectories[start - 1 : end + 1, id - 1]
            else:
                where = self.trajectories[start : end + 1, id - 1]
        self.go_to_error.emit(kind, start, end, where, id)

    def set_references(
        self, traj: np.ndarray, all_identified: np.ndarray, duplicated: np.ndarray
    ):
        self.trajectories = traj
        self.all_identified = all_identified
        self.duplicated = duplicated
        self.update_btn.click()

    def getErrors(self) -> list[tuple[str, int, int, int]]:
        missing_id_err = get_list_of_False_for_id(~np.isnan(self.trajectories[..., 0]))
        centroid_wo_id_err = get_list_of_False(self.all_identified)
        duplicated_id_err = get_list_of_False_for_id(~self.duplicated)
        impossible_jumps_err = get_impossible_jumps(
            self.trajectories, self.long_jumps_th.value()
        )
        # TODO Add more errors (super-crossings)
        return (
            [("Miss id",) + err for err in missing_id_err]
            + [("No Id", -1) + err for err in centroid_wo_id_err]
            + [("Dupl",) + err for err in duplicated_id_err]
            + [("Jump",) + err for err in impossible_jumps_err]
        )

    def update_list_of_errors(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for error in self.getErrors():
            self.table.insertRow(0)
            start, end = error[-2:]
            for i, item in enumerate(error + (end - start,)):
                self.table.setItem(0, i, CustomTableWidgetItem(item))
        self.left_label.setText(f"List of errors ({self.table.rowCount()} errors)")
        self.table.setSortingEnabled(True)


def get_list_of_False(arr: np.ndarray) -> list[tuple[int, int]]:
    dif = np.diff(np.concatenate(([-1], np.where(arr)[0], [len(arr)])))
    end = np.cumsum(dif) - 1
    nan = dif - 1
    valid = nan > 0
    return [(e - n, e) for e, n in zip(end[valid], nan[valid])]


def get_list_of_False_for_id(data: np.ndarray) -> list[tuple[int, int, int]]:
    return [
        (fish_id + 1,) + tuples
        for fish_id in range(data.shape[1])
        for tuples in get_list_of_False(data[:, fish_id])
    ]


def get_impossible_jumps(traj: np.ndarray, sigma: float = 4.0):
    speed = np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=-1))
    mean, std = np.nanmean(speed), np.nanstd(speed)
    speed[np.isnan(speed)] = 0
    speed = np.row_stack((np.zeros(speed.shape[1]), speed))
    accepted_speed = speed < (mean + sigma * std)
    return get_list_of_False_for_id(accepted_speed)
