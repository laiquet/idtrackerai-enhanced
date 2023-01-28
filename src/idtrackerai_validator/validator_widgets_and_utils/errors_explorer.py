import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ErrorsExplorer(QWidget):
    go_to_error = pyqtSignal(str, int, int)  # type, id, start

    def __init__(self):
        super().__init__()
        self.table = QTableWidget(1, 5)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(["Type", "Id", "Start", "End", "Length"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setMinimumSectionSize(10)
        self.table.currentCellChanged.connect(self.cell_clicked)

        self.info_widget = QListWidget()
        self.info_widget.setAlternatingRowColors(True)

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
        self.setLayout(layout)

    def cell_clicked(self, row: int, col: int):
        if row < 0 or col < 0:
            return
        type, id, start, end, length = self.table.item(row, col).data(
            Qt.ItemDataRole.UserRole
        )
        self.go_to_error.emit(type, id, start)

    def setTrajectories(self, traj: np.ndarray):
        self.trajectories = traj
        self.update_btn.click()

    def getErrors(self) -> list[tuple[str, int, int, int, int]]:
        nan_errs = get_list_of_nans_from_traj(self.trajectories)
        # Add more errors (duplicated ids, impossible jumps, super-crossings)
        return [("Nan",) + err for err in nan_errs]

    def update_list_of_errors(self):
        self.table.setRowCount(0)
        for error in self.getErrors():
            self.table.insertRow(0)
            for i, item in enumerate(error):
                widgetItem = QTableWidgetItem(str(item))
                widgetItem.setData(Qt.ItemDataRole.UserRole, error)
                widgetItem.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(0, i, widgetItem)

        self.left_label.setText(f"List of errors ({self.table.rowCount()} errors)")


def find_nans_1D(data: np.ndarray) -> list[tuple[int, int, int]]:
    """Returns a list of the nans location for a 1D array:
        (fish_id (optional), start, end, length of slice)
    In such a way that data[start] is the first nan and data[end-1] is the last nan
    """
    assert (
        data.ndim == 1
    ), f"Only one dimensional arrays, the given array has shape {data.shape}"

    dif = np.diff(np.concatenate(([-1], np.where(~np.isnan(data))[0], [len(data)])))
    end = np.cumsum(dif) - 1
    nan = dif - 1
    valid = nan > 0
    return [(e - n, e, n) for e, n in zip(end[valid], nan[valid])]


def get_list_of_nans_from_traj(traj: np.ndarray):
    """Returns a list of the nans location for a trajectory 2D/3D array:
        (fish_id (only for 3D arrays), start, end, length of slice)
    In such a way that traj[fish_id, start] is the first nan and
    traj[fish_id, end-1] is the last nan.
    Elements in the list are sorted by length of slice (reverse)
    """
    if traj.ndim == 3:
        traj = traj[..., 0]

    nans: list[tuple[int, int, int, int]] = []
    for fish_id in range(traj.shape[1]):
        nans += [(fish_id,) + tuples for tuples in find_nans_1D(traj[:, fish_id])]

    return nans
