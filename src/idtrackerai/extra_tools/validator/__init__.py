import sys
from argparse import ArgumentParser
from pathlib import Path

from qtpy.QtWidgets import QApplication

from idtrackerai.utils import manage_exception, wrap_entrypoint

from .validation_GUI import ValidationGUI


@wrap_entrypoint
def idtrackerai_validate_entrypoint() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "session_directory", help="Session directory to validate", type=Path, nargs="?"
    )
    args = parser.parse_args()

    idtrackerai_validate(args.session_directory)


def idtrackerai_validate(session_directory: Path | None) -> None:
    # this catches exceptions when raised inside Qt
    def excepthook(exc_type, exc_value, exc_tb) -> None:
        assert QApplication  # Pylance is happier with this
        QApplication.quit()
        manage_exception(exc_value)

    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    window = ValidationGUI(session_directory)
    window.show()
    app.exec()
