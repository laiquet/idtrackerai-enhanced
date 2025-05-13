"""Module to validate the tracking data of a session, see :ref:`validator`."""

import sys
from argparse import ArgumentParser
from multiprocessing import Process
from pathlib import Path

from qtpy.QtWidgets import QApplication

from idtrackerai import Session
from idtrackerai.utils import (
    LOGGING_QUEUE,
    manage_exception,
    setup_logging_queue,
    wrap_entrypoint,
)

from .validation_GUI import ValidationGUI


@wrap_entrypoint
def idtrackerai_validate_entrypoint() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "session_directory", help="Session directory to validate", type=Path, nargs="?"
    )
    args = parser.parse_args()

    idtrackerai_validate(args.session_directory)


def idtrackerai_validate(session_directory: Session | Path | str | None = None) -> None:
    """Initialize the Validator GUI, called by the command ``idtrackerai_validate``.

    .. seealso::
        Documentation for :ref:`validator`

    Parameters
    ----------
    session_directory : Session | Path | str | None, optional
        Session or path to the Session folder to validate. If None, a blank Validator is opened.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        run_gui(session_directory, None)
    else:
        # On Linux, we run the GUI in a separate process to catch libxcb errors
        p = Process(target=run_gui, args=(session_directory, LOGGING_QUEUE))
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"QApplication crashed with exit code {p.exitcode}")


def run_gui(session_directory, logging_queue):
    if logging_queue:
        setup_logging_queue(logging_queue)

    # this catches exceptions when raised inside Qt
    def excepthook(exc_type, exc_value, exc_tb) -> None:
        QApplication.quit()
        manage_exception(exc_value)

    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    window = ValidationGUI(session_directory)
    window.show()
    app.exec()
