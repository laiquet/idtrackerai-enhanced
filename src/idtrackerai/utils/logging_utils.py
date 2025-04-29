"Functions to manage logging and exception handling"

import logging
import logging.handlers
import multiprocessing
import os
import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from functools import wraps
from importlib import metadata
from pathlib import Path
from platform import platform, python_version
from shutil import copyfileobj
from tempfile import TemporaryFile
from traceback import extract_tb

from packaging.version import Version
from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler

from .py_utils import IdtrackeraiError, resolve_path
from .telemetry import check_version_on_console_thread, report_usage_on_console_thread

# The logging from other processes is sent to the listener thread of
# the main process by using the LOGGING_QUEUE.
# This queue is created in the main process and passed to the other processes.
if multiprocessing.current_process().name == "MainProcess":
    LOGGING_QUEUE = multiprocessing.Queue()
else:
    LOGGING_QUEUE = None

LOG_FILE_PATH = resolve_path("idtrackerai.log")

# This temporary file stores the log until the tracking is over
# and its contents are copied into the session folder log file
TMP_LOG_FILE = TemporaryFile("w+", suffix=".idtrackerai.log")

ERROR_MSG = (
    "\n\nIf this error happens right after the installation,"
    " check our installation troubleshooting guide"
    " https://idtracker.ai/latest/install/installation_troubleshooting.html"
    "\n\nIf this error persists please let us know by following any of the following"
    " options:\n  - Posting on https://groups.google.com/g/idtrackerai_users\n  -"
    " Opening an issue at https://gitlab.com/polavieja_lab/idtrackerai\n  - Sending an"
    " email to info@idtracker.ai\nShare the log file ({}) when doing"
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


def _log_listener_thread(queue: multiprocessing.Queue) -> None:
    logger = logging.getLogger()
    while True:
        record = queue.get()
        if record is None:
            # We send this as a sentinel to tell the listener to quit.
            break
        logger.handle(record)


def setup_logging_queue(queue: multiprocessing.Queue) -> None:
    """
    Intended to be run in a child process. Sets up a logging queue
    to send log messages to the listener thread in the main process.
    """
    if queue is None:
        # This is the main process, nothing to do
        return
    logging_queue_handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    if any(isinstance(handler, LevelRichHandler) for handler in root.handlers):
        # When multiprocessing with 'fork' method, the child process inherits the parent's handlers
        # We do not want to add the handler twice in this case
        return
    root.handlers.clear()
    root.addHandler(logging_queue_handler)


def init_logger(level: int = logging.DEBUG, write_to_disk: bool = False) -> None:
    """Initializes the logger with a custom re-implementation of Rich logging handler"""
    global LOG_FILE_PATH
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

    # The first handler is the terminal, the second one the .log file,

    handlers = [
        LevelRichHandler(
            console=Console(width=size), rich_tracebacks=True, show_level=False
        )
    ]

    if write_to_disk:
        for counter in range(1, 21):  # try 30 different log file names...
            try:
                LOG_FILE_PATH.unlink(True)  # avoid conflicts and merged files
            except PermissionError as exc:
                # In Windows this arises when the log file is open by another session
                logging.warning(
                    f"Could not set up the logging file {LOG_FILE_PATH} because "
                    f"of the following PermissionError: {exc}"
                )
                LOG_FILE_PATH = LOG_FILE_PATH.with_stem(f"idtrackerai_{counter}")
            else:
                break
        else:
            logging.error("Could not set up the logging file")

        handlers.extend(
            (
                LevelRichHandler(
                    console=Console(
                        file=LOG_FILE_PATH.open("w", encoding="utf_8"),  # noqa SIM115
                        width=logger_width_when_no_terminal,
                    ),
                    show_level=False,
                ),
                LevelRichHandler(
                    console=Console(
                        file=TMP_LOG_FILE, width=logger_width_when_no_terminal
                    ),
                    show_level=False,
                ),
            )
        )

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=handlers,
    )

    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # avoid huge logs with corrupted videos
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.captureWarnings(True)
    logging.info(
        f"[bold]Welcome to idtracker.ai[/] {metadata.version('idtrackerai')}",
        extra={"markup": True},
    )
    logging.debug(
        f"Date: {str(datetime.now()).split('.')[0]}\n"
        f"Running Python '{python_version()}' in '{platform(True)}'"
    )
    if write_to_disk:
        logging.info("Writing log in %s", LOG_FILE_PATH)
    else:
        logging.info("Not writing log in any file")
    logging.info("Using NumPy %s", metadata.version("numpy"))

    try:
        logging.info(
            "Using headless OpenCV %s", metadata.version("opencv-python-headless")
        )
    except metadata.PackageNotFoundError:
        logging.info("Headless OpenCV not found")

    try:
        logging.info("Using regular OpenCV %s", metadata.version("opencv-python"))
    except metadata.PackageNotFoundError:
        logging.info("Regular OpenCV not found")

    logging.info(
        'Using "%s" multiprocessing start method', multiprocessing.get_start_method()
    )

    if LOGGING_QUEUE:
        # We start the listener thread which will receive log messages from other processes
        threading.Thread(target=_log_listener_thread, args=(LOGGING_QUEUE,)).start()


def wrap_entrypoint(main_function: Callable):
    """Function decorator for CLI entry points.
    It initializes the logger, checks the software version and manages possible Exceptions
    """

    @wraps(main_function)
    def ret_fun(*args, **kwargs):
        init_logger(write_to_disk=True)
        check_version_on_console_thread()
        report_usage_on_console_thread()
        try:
            return_value = main_function(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            manage_exception(exc)
            if hasattr(exc, "log_path"):
                TMP_LOG_FILE.flush()
                TMP_LOG_FILE.seek(0)
                with open(exc.log_path, "w") as file:  # type: ignore
                    copyfileobj(TMP_LOG_FILE, file)
                logging.info(f"Log file copied to {exc.log_path}")  # type: ignore
            return_value = False
        finally:
            if LOGGING_QUEUE:
                # We send this as a sentinel to tell the listener thread to quit.
                LOGGING_QUEUE.put(None)
        return return_value

    return ret_fun


def manage_exception(exc: BaseException) -> None:
    """Prints useful log messages depending on the type of Exception"""
    ERROR_MSG_ = ERROR_MSG.format(LOG_FILE_PATH)
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
            logging.info(ERROR_MSG_)

        case RuntimeError():
            with suppress(metadata.PackageNotFoundError):
                if (Version(metadata.version("torch")) < Version("2.3")) and (
                    Version(metadata.version("numpy")) >= Version("2.0")
                ):
                    logging.error(str(exc))
                    logging.critical(
                        "This error may be caused by your PyTorch installation (version %s) being incompatible with NumPy 2.0 or higher, please update PyTorch by running the installation command in https://pytorch.org/get-started/locally/#start-locally",
                        metadata.version("torch"),
                    )
                    return

            logging.critical("%s: %s", type(exc).__name__, exc, exc_info=exc)
            logging.info(ERROR_MSG_)

        case Exception():
            logging.critical("%s: %s", type(exc).__name__, exc, exc_info=exc)
            logging.info(ERROR_MSG_)
