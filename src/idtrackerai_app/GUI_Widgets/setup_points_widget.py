from PyQt6.QtWidgets import QInputDialog
from .list_layout import List_Layout
import re
from PyQt6.QtCore import Qt


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = re.compile("[@!$%^&*?/\|~:]")
    return not regex.search(string) == None


class SetupPointsWidget(List_Layout):
    def __init__(self):
        super().__init__()
        self.CheckBox.setText("Setup Points")
        self.add.clicked.connect(self.add_clicked)
        self.setup_points_dict = {}

        self.CheckBox.stateChanged.connect(self.CheckBox_changed_visible)
        self.ListChanged.connect(self.update_legend)

    def CheckBox_changed_visible(self, enabled):
        for i in range(self.list.count()):
            name = self.list.item(i).text().split(":")[0]
            self.setup_points_dict[name].set(visible=enabled)

    def add_clicked(self, checked):
        if checked:
            existing_names = [
                self.list.item(i).text().split(":")[0]
                for i in range(self.list.count())
            ]

            invalid = True
            dialog_text = "Enter setup points name:"
            while invalid:
                name, ok = QInputDialog.getText(
                    self.add, "idtracker.ai", dialog_text
                )
                if not ok:
                    self.add.setChecked(False)
                    break
                name = name.strip()
                invalid = has_invalid_chars(name)
                if invalid:
                    dialog_text = "Invalid characters encountered"
                if name in existing_names:
                    invalid = True
                    dialog_text = "Enter a non existing name"

            self.setup_name = name

        else:
            xy = self.plot_line.get_xydata().astype(int)
            self.plot_line.set_data([], [])

            self.setup_points_dict[self.setup_name] = self.ax.plot(
                *xy.T, ".", label=self.setup_name
            )[0]
            self.add_str_to_list(
                self.setup_name + ": " + str(xy).replace("\n", ",")
            )

    def update_legend(self):
        legend_needed = self.CheckBox.isChecked() and self.list.count()
        if legend_needed:
            self.ax.legend()
        else:
            self.ax.legend([]).set_visible(False)

    def remove_item(self):
        item = self.list.itemAt(self.sender().parent().pos())
        self.setup_points_dict.pop(
            item.data(Qt.UserRole).split(":")[0]
        ).remove()
        self.list.takeItem(self.list.row(item))
