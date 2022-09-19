from PyQt6.QtWidgets import (
    QInputDialog,
)

from .list_layout import List_Layout
import re


def has_invalid_chars(string):
    if len(string) == 0:
        return True
    regex = re.compile("[@!$%^&*?/\|~:]")
    return not regex.search(string) == None


class setup_line:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class SetupPointsWidget(List_Layout):
    def __init__(self):
        super().__init__()
        self.CheckBox.setText("Setup Points")
        self.add.clicked.connect(self.add_clicked)

        # self.ROI_popup = SetupPointsPopUp()
        self.setup_points_dict = {}

        self.CheckBox.stateChanged.connect(self.CheckBox_changed_visible)

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
            self.list.addItem(
                self.setup_name + ": " + str(xy).replace("\n", ",")
            )

    def remove_event(self):
        for item in self.list.selectedItems():
            self.setup_points_dict.pop(item.text().split(":")[0]).remove()
            self.list.takeItem(self.list.row(item))
        if not len(self.list.selectedItems()):
            self.remove.setEnabled(False)
