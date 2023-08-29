import sys
from argparse import ArgumentParser
from pathlib import Path

from qtpy.QtWidgets import QApplication

from idtrackerai.utils import wrap_entrypoint
from idtrackerai_validator.validation_GUI import ValidationGUI


def input_args():
    parser = ArgumentParser()
    parser.add_argument(
        "session_directory", help="Session directory to validate", type=Path, nargs="?"
    )
    return parser.parse_args()


@wrap_entrypoint
def main():
    args = input_args()
    app = QApplication(sys.argv)
    window = ValidationGUI(args.session_directory)
    window.show()
    app.exec()
