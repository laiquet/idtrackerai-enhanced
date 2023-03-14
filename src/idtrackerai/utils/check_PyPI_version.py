import json
import logging
from threading import Thread
from urllib.request import urlopen

import idtrackerai


def check_version_on_console_thread():
    Thread(target=check_version_on_console).start()


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
            all_versions: dict = json.load(json_data)["releases"]
    except Exception:
        return False, "Could not reach PyPI website to check for updates"

    non_yanked_versions = (
        name for name, properties in all_versions.items() if not properties[0]["yanked"]
    )

    stable_versions = filter(
        lambda v: v.replace(".", "").isdigit(), non_yanked_versions
    )

    last_version = tuple(stable_versions)[-1]  # the newest version

    current_version = idtrackerai.__version__
    if available_is_greater(last_version, current_version):
        return (
            True,
            (
                f"A new release of idtracker.ai available: {current_version} ->"
                f"{last_version}\n"
                "To update, run: python3 -m pip install --upgrade idtrackerai"
            ),
        )

    return (
        False,
        (
            "There are currently no updates available.\n"
            f"Current idtrackerai version: {current_version}"
        ),
    )
