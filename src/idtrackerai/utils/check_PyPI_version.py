import json
import logging
from typing import Iterable
from urllib.request import urlopen

import idtrackerai


def available_is_greater(available: str, current: str):
    available_parts = available.split(".")
    current_parts = current.split(".")

    for available_part, current_part in zip(available_parts, current_parts):
        if available_part > current_part:
            return True
        if available_part < current_part:
            return False
    return False


def check_version_on_console():
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.INFO)
    try:
        warn, message = check_version()
    finally:
        logger.setLevel(old_level)

    if warn:
        logging.warning(message)


def check_version() -> tuple[bool, str]:
    try:
        with urlopen("https://pypi.org/pypi/idtrackerai/json") as json_data:
            all_versions: Iterable[str] = json.load(json_data)["releases"].keys()
    except Exception:
        return True, "Could not reach PyPI website to check for updates"

    stable_versions = filter(lambda v: v.replace(".", "").isdigit(), all_versions)

    last_version = tuple(stable_versions)[-1]  # the newest version

    current_version = idtrackerai.__version__
    if available_is_greater(last_version, current_version):
        return True, (
            f"A new release of idtracker.ai available: {current_version} ->"
            f"{last_version}\n"
            "To update, run: python3 -m pip install --upgrade idtrackerai"
        )

    return False, (
        "There are currently no updates available.\n"
        f"Current idtrackerai version: {current_version}"
    )
