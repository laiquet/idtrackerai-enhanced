import sys
from argparse import ArgumentParser
from pathlib import Path

from qtpy.QtWidgets import QApplication

from idtrackerai.utils import manage_exception, wrap_entrypoint
from idtrackerai.validator.validation_GUI import ValidationGUI


def input_args():
    parser = ArgumentParser()
    parser.add_argument(
        "session_directory", help="Session directory to validate", type=Path, nargs="?"
    )
    return parser.parse_args()


@wrap_entrypoint
def main():
    # this catches exceptions when raised inside Qt
    def excepthook(exc_type, exc_value, exc_tb):
        assert QApplication  # Pylance is happier with this
        QApplication.quit()
        manage_exception(exc_value)

    sys.excepthook = excepthook

    args = input_args()
    app = QApplication(sys.argv)
    window = ValidationGUI(args.session_directory)
    window.show()
    app.exec()
