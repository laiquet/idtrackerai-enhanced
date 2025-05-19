"""Module to report usage analytics and check for updates."""

import json
import logging
import os
import re
import sys
from contextlib import suppress
from datetime import datetime
from importlib import metadata
from itertools import zip_longest
from pathlib import Path
from platform import platform, python_version
from threading import Thread
from urllib.request import urlopen

from requests import post

ANALYTICS_STATE_FILE_PATH = Path(__file__).parent / "usage_analytics_state.json"
ANALYTICS_URL = "https://analytics.polaviejalab.org/report_usage.php"
PYPI_URL = "https://pypi.org/simple/idtrackerai"
ANALYTICS_ENVIRON = "IDTRACKERAI_DISABLE_ANALYTICS"


def check_version_on_console_thread() -> None:
    Thread(target=check_version_on_console).start()


def report_usage_on_console_thread() -> None:
    Thread(target=report_usage).start()


def set_usage_analytics_state(enabled: bool) -> None:
    ANALYTICS_STATE_FILE_PATH.write_text(json.dumps(enabled))


def get_usage_analytics_state() -> bool:
    """Returns the current state of the usage
    analytics reporting. If the state is not set,
    it will return True and set the state to True."""
    environ = os.environ.get(ANALYTICS_ENVIRON, "").lower()
    if environ in ("1", "true"):
        state = True
    elif environ in ("0", "false"):
        state = False
    elif not ANALYTICS_STATE_FILE_PATH.exists():
        state = True
    else:
        state = json.loads(ANALYTICS_STATE_FILE_PATH.read_text())

    current_state = (
        json.loads(ANALYTICS_STATE_FILE_PATH.read_text())
        if ANALYTICS_STATE_FILE_PATH.exists()
        else None
    )
    if state != current_state:
        set_usage_analytics_state(state)

    return state


def report_usage() -> None:
    """Reports usage analytics to the server."""
    usage_analytics_enabled = get_usage_analytics_state()
    if not usage_analytics_enabled:
        logging.info("Usage analytics reporting is disabled")
        return

    try:
        response = post(
            ANALYTICS_URL,
            json={
                "date": datetime.now().astimezone().isoformat(),
                "platform": platform(True),
                "idtrackerai_version": metadata.version("idtrackerai"),
                "python_version": python_version(),
                "command": sys.argv,
            },
        )
        if response.status_code != 200:
            logging.error(
                f"Error reporting usage analytics. Status code: {response.status_code} {response.text}"
            )
    except Exception as e:
        logging.error(f"Error reporting usage analytics: {e}")


def _available_is_greater(available: str, current: str) -> bool:
    for available_part, current_part in zip_longest(
        map(int, available.split(".")), map(int, current.split(".")), fillvalue=0
    ):
        if available_part > current_part:
            return True
        if available_part < current_part:
            return False
    return False


def _available_is_equal(available: str, current: str) -> bool:
    for available_part, current_part in zip_longest(
        map(int, available.split(".")), map(int, current.split(".")), fillvalue=0
    ):
        if available_part > current_part:
            return False
        if available_part < current_part:
            return False
    return True


def check_version_on_console() -> None:
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.INFO)
    with suppress(Exception):
        warn, message = check_version()
        if warn:
            logging.warning(message)
    logger.setLevel(old_level)


def check_version() -> tuple[bool, str]:
    """Check if there is a new version of idtracker.ai available."""
    try:
        out_text = urlopen(PYPI_URL, timeout=10).read().decode("utf-8")
    except Exception:
        return False, "Could not reach PyPI website to check for updates"

    if not isinstance(out_text, str) or not out_text:
        return False, "Error getting web text"

    # TODO maybe use from html.parser import HTMLParser?
    no_yanked_versions = "\n".join(
        line for line in out_text.splitlines() if "yanked" not in line
    )
    versions: list[tuple[str, str]] = re.findall(
        ">idtrackerai-(.+?)(.tar.gz|-py3-none-any.whl)<", no_yanked_versions
    )

    current_version = metadata.version("idtrackerai").split("a")[0]

    is_current_version_alpha = "a" in metadata.version("idtrackerai")
    for version, _file_extension in versions[::-1]:
        if not version.replace(".", "").isdigit():
            continue  # not a stable version

        if _available_is_greater(version, current_version):
            return (
                True,
                f"A new release of idtracker.ai is available: {current_version} -> "
                f"{version}\n"
                'To update, run: "python -m pip install --upgrade idtrackerai"',
            )
        elif is_current_version_alpha and _available_is_equal(version, current_version):
            return (
                True,
                "You are running an alpha version of idtracker.ai and the stable"
                f" version is available: {metadata.version('idtrackerai')} ->"
                f" {version}\nTo update, run: python -m pip install --upgrade"
                " idtrackerai",
            )

    return (
        False,
        "There are currently no updates available.\n"
        f"Current idtrackerai version: {current_version}",
    )
