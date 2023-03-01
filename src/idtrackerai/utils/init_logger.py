import logging
import os
import sys
from importlib import metadata
from platform import platform
from threading import Thread

from rich.console import Console
from rich.logging import RichHandler

from .check_PyPI_version import check_version_on_console


def initLogger(testing=False, check_version=True):
    logger_width_when_no_terminal = 150
    try:
        os.get_terminal_size()
    except OSError:
        # stdout is sent to file. We define logger width to a constant
        size = logger_width_when_no_terminal
    else:
        # stdout is sent to terminal
        # We define logger width to adapt to the terminal width
        size = None

    # The first handler is the terminal, the second one the .log file,
    # both rendered with Rich and full logging (level=0)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="%H:%M:%S",
        force=not testing,
        handlers=[
            RichHandler(console=Console(width=size)),
            RichHandler(
                console=Console(
                    file=open("idtrackerai.log", "w", encoding="utf_8"),
                    width=logger_width_when_no_terminal,
                )
            ),
        ],
    )

    logging.getLogger("PyQt6").setLevel(logging.INFO)
    logging.info("Welcome to idtracker.ai")
    logging.debug(
        f"Running idTracker.ai '{metadata.version('idtrackerai')}'"
        f" on Python '{sys.version.split(' ')[0]}'\nPlatform: '{platform(True)}'"
    )

    if check_version:
        Thread(target=check_version_on_console).start()
