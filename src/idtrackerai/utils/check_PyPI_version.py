import logging

import requests

import idtrackerai


def available_is_greater(available: str, current: str):
    available_parts = available.split(".")
    current_parts = current.split(".")

    for available_part, current_part in zip(available_parts, current_parts):
        if available_part > current_part:
            return True
        elif available_part < current_part:
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
        response = requests.get("https://pypi.org/pypi/idtrackerai/json", timeout=2)
    except requests.exceptions.RequestException:
        return True, "Could not reach PyPI website to check for updates"
    if response.status_code != 200:
        return True, "Could not reach PyPI website to check for updates"

    data = response.json()

    versions: list[str] = list(data["releases"].keys())

    for available_version in versions[::-1]:
        if available_version.replace(".", "").isdigit():
            break

    current_version = idtrackerai.__version__
    if available_is_greater(available_version, current_version):
        return True, (
            f"A new release of idtracker.ai available: {current_version} ->"
            f"{available_version}\n"
            "To update, run: python3 -m pip install --upgrade idtrackerai"
        )

    else:
        return False, (
            "There are currently no updates available.\n"
            f"Current idtrackerai version: {current_version}"
        )
