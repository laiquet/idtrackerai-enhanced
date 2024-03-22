import logging
import os
from datetime import datetime
from functools import wraps
from importlib import metadata
from pathlib import Path
from platform import platform, python_version
from traceback import extract_tb
from typing import Callable

from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler

from .check_PyPI_version import check_version_on_console_thread
from .py_utils import IdtrackeraiError, resolve_path

LOG_FILE_PATH = resolve_path("idtrackerai.log")

ERROR_MSG = (
    "\n\nIf this error happens right after the installation,"
    " check our installation troubleshooting guide"
    " https://idtracker.ai/latest/install/installation_troubleshooting.html"
    "\n\nIf this error persists please let us know by following any of the following"
    " options:\n  - Posting on https://groups.google.com/g/idtrackerai_users\n  -"
    " Opening an issue at https://gitlab.com/polavieja_lab/idtrackerai\n  - Sending an"
    f" email to idtrackerai@gmail.com\nShare the log file ({LOG_FILE_PATH}) when doing"
    " any of the options above"
)

LEVEL_FORMAT = {
    "WARNING": "red",
    "WARN": "red",
    "ERROR": "red bold",
    "FATAL": "red bold reverse",
    "CRITICAL": "red bold reverse",
    "NOTSET": "red",
}


class LevelRichHandler(RichHandler):
    """Modified Rich Logging Handler that prints the logging level only if its
    equal or grater than a warning. Intended to work with `show_level=False`"""

    def render_message(
        self, record: logging.LogRecord, message: str
    ) -> ConsoleRenderable:
        """Method override to manage log level only if its >= warning

        Args:
            record (LogRecord): logging Record.
            message (str): String containing log message.

        Returns:
            ConsoleRenderable: Renderable to display log message.
        """
        if record.levelno >= logging.WARNING:
            record.markup = True
            format = LEVEL_FORMAT.get(record.levelname, "red")
            message = f"[{format}]{record.levelname}[/{format}] {message}"
        return super().render_message(record, message)


def initLogger(level: int = logging.DEBUG):
    logger_width_when_no_terminal = 126
    try:
        os.get_terminal_size()
    except OSError:
        # stdout is sent to file. We define logger width to a constant
        size = logger_width_when_no_terminal
    else:
        # stdout is sent to terminal
        # We define logger width to adapt to the terminal width
        size = None

    LOG_FILE_PATH.unlink(True)  # avoid conflicts and merged files

    # The first handler is the terminal, the second one the .log file,
    # both rendered with Rich and full logging (level=0)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            LevelRichHandler(
                console=Console(width=size), rich_tracebacks=True, show_level=False
            ),
            LevelRichHandler(
                console=Console(
                    file=LOG_FILE_PATH.open("w", encoding="utf_8"),  # noqa SIM115
                    width=logger_width_when_no_terminal,
                ),
                show_level=False,
            ),
        ],
    )

    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # avoid huge logs with corrupted videos
    logging.captureWarnings(True)
    logging.info("[bold]Welcome to idtracker.ai", extra={"markup": True})
    logging.debug(
        f"Running idtracker.ai '{metadata.version('idtrackerai')}'"
        f" on Python '{python_version()}'\nPlatform: '{platform(True)}'"
        "\nDate: " + str(datetime.now()).split(".")[0]
    )
    logging.info("Writing log in %s", LOG_FILE_PATH)


def wrap_entrypoint(main_function: Callable):
    @wraps(main_function)
    def ret_fun(*args, **kwargs):
        initLogger()
        check_version_on_console_thread()
        try:
            return main_function(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            manage_exception(exc)
            return False

    return ret_fun


def manage_exception(exc: BaseException):
    match exc:
        case IdtrackeraiError():
            tb = extract_tb(exc.__traceback__)[-1]
            logging.critical(
                "%s [bright_black](from %s:%d)[/]",
                exc,
                Path(*Path(tb.filename).parts[-2:]),
                tb.lineno,
                extra={"markup": True},
            )
        case KeyboardInterrupt():
            logging.critical("KeyboardInterrupt", exc_info=False)
            return
        case ModuleNotFoundError():
            if "torch" in str(exc):
                logging.critical(
                    "Module PyTorch is not installed, follow their guideline to install"
                    " it (https://pytorch.org/get-started/locally/). Original"
                    ' exception: "%s"',
                    exc,
                )
                return
            logging.critical("%s: %s", type(exc).__name__, exc, exc_info=exc)
            logging.info(ERROR_MSG)
            return
        case Exception():
            logging.critical("%s: %s", type(exc).__name__, exc, exc_info=exc)
            logging.info(ERROR_MSG)
            return
