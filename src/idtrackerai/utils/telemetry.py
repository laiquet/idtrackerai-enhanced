"""Module to report usage analytics and check for updates."""

import json
import logging
import os
import re
import sys
from contextlib import suppress
from datetime import datetime
from enum import Enum
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from platform import platform, python_version
from threading import Thread
from urllib.request import urlopen

from requests import post

ANALYTICS_STATE_FILE_PATH = Path(__file__).parent / "usage_analytics_state.json"
ANALYTICS_URL = "https://analytics.polaviejalab.org/report_usage.php"
PYPI_URL = "https://pypi.org/simple/idtrackerai"
ANALYTICS_ENVIRON = "IDTRACKERAI_DISABLE_ANALYTICS"


class ComparisonResult(Enum):
    EQUAL = "equal"
    MAJOR_UPDATE = "major_update"
    MINOR_UPDATE = "minor_update"
    PATCH_UPDATE = "patch_update"
    STABLE_RELEASE = "stable_release"
    ERROR = "error"

    @property
    def update_available(self) -> bool:
        return self in {
            ComparisonResult.MAJOR_UPDATE,
            ComparisonResult.MINOR_UPDATE,
            ComparisonResult.PATCH_UPDATE,
        }


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
        state = False
    elif environ in ("0", "false") or not ANALYTICS_STATE_FILE_PATH.exists():
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


def check_version_on_console() -> None:
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.INFO)
    with suppress(Exception):
        kind, message = check_version()
        if kind == ComparisonResult.ERROR:
            logging.error(message)
        elif kind != ComparisonResult.EQUAL:
            logging.warning(message)
    logger.setLevel(old_level)


@lru_cache(maxsize=1)
def check_version() -> tuple[ComparisonResult, str]:
    """Check if there is a new version of idtracker.ai available."""
    try:
        out_text = urlopen(PYPI_URL, timeout=5).read().decode("utf-8")
    except Exception:
        return ComparisonResult.ERROR, "Error fetching PyPI data"

    if not isinstance(out_text, str) or not out_text:
        return ComparisonResult.ERROR, "Error reading PyPI data"

    no_yanked_versions = "\n".join(
        line for line in out_text.splitlines() if "yanked" not in line
    )
    matches: list[tuple[str, str]] = re.findall(
        ">idtrackerai-(.+?)(.tar.gz|-py3-none-any.whl)<", no_yanked_versions
    )

    current_version_str = metadata.version("idtrackerai")
    current_version = current_version_str.split("a")[0]
    try:
        current_version = tuple(map(int, current_version.split(".")))
    except Exception as e:
        logging.error(f"Error parsing current version: {e}")
        return ComparisonResult.ERROR, (
            f"The current version of idtracker.ai ({current_version}) is not a "
            "valid version string."
        )

    latest_version = max(
        tuple(map(int, version.split(".")))
        for version, _file_extension in matches
        if version.replace(".", "").isdigit()  # only keep stable versions
    )

    latest_version_str = ".".join(map(str, latest_version))

    try:
        if latest_version[0] > current_version[0]:
            return ComparisonResult.MAJOR_UPDATE, (
                f"A new major release of idtracker.ai is available: {current_version_str} -> "
                f"{latest_version_str}\n"
                'To update, run: "python -m pip install --upgrade idtrackerai"'
            )
        elif latest_version[1] > current_version[1]:
            return ComparisonResult.MINOR_UPDATE, (
                f"A new minor release of idtracker.ai is available: {current_version_str} -> "
                f"{latest_version_str}\n"
                'To update, run: "python -m pip install --upgrade idtrackerai"'
            )
        elif latest_version[2] > current_version[2]:
            return ComparisonResult.PATCH_UPDATE, (
                f"A new patch release of idtracker.ai is available: {current_version_str} -> "
                f"{latest_version_str}\n"
                'To update, run: "python -m pip install --upgrade idtrackerai"'
            )
        elif "a" in current_version_str:
            return ComparisonResult.STABLE_RELEASE, (
                "You are running an alpha version of idtracker.ai and the stable"
                f" version is available: {metadata.version('idtrackerai')} ->"
                f" {latest_version_str}\nTo update, run: python -m pip install --upgrade"
                " idtrackerai"
            )
        else:
            return ComparisonResult.EQUAL, (
                "There are currently no updates available.\n"
                f"Current idtrackerai version: {current_version_str}"
            )
    except IndexError as exc:
        return ComparisonResult.ERROR, (
            f"Error comparing versions {latest_version_str} and {current_version}: {exc}"
        )
