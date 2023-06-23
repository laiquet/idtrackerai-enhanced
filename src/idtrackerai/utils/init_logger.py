import logging
import os
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from platform import platform
from typing import Callable

from rich.console import Console
from rich.logging import RichHandler

from .check_PyPI_version import check_version_on_console_thread

LOG_FILE_PATH = Path("idtrackerai.log").resolve()


class CustomError(Exception):
    pass


def initLogger(check_version=True, level: int = logging.DEBUG):
    logger_width_when_no_terminal = 130
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
            RichHandler(console=Console(width=size)),
            RichHandler(
                console=Console(
                    file=LOG_FILE_PATH.open("w", encoding="utf_8"),  # noqa SIM115
                    width=logger_width_when_no_terminal,
                )
            ),
        ],
    )

    logging.captureWarnings(True)
    logging.info("Welcome to idtracker.ai")
    logging.debug(
        f"Running idtracker.ai '{metadata.version('idtrackerai')}'"
        f" on Python '{sys.version.split(' ')[0]}'\nPlatform: '{platform(True)}'"
        "\nDate: "
        + str(datetime.now()).split(".")[0]
    )
    logging.info("Writing log in %s", LOG_FILE_PATH)

    if check_version:
        check_version_on_console_thread()


def wrap_exceptions(main_function: Callable):
    def applicator(*args, **kwargs):
        try:
            return main_function(*args, **kwargs)
        except CustomError as error:
            logging.critical(error, exc_info=False)
            return False
        except Exception as error:
            logging.critical(error, exc_info=True)
            logging.warning(
                (
                    "\n\nIf this error persists please let us know by "
                    "following any of the following options:\n"
                    "  - posting on "
                    "https://groups.google.com/g/idtrackerai_users\n"
                    "  - opening an issue at "
                    "https://gitlab.com/polavieja_lab/idtrackerai\n"
                    "  - sending an email to idtrackerai@gmail.com\n"
                    "Share the log file (%s) when "
                    "doing any of the options above"
                ),
                LOG_FILE_PATH,
            )
            return False

    return applicator
