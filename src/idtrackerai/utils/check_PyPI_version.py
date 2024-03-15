import logging
import re
from contextlib import suppress
from importlib import metadata
from itertools import zip_longest
from threading import Thread
from urllib.request import urlopen


def check_version_on_console_thread():
    Thread(target=check_version_on_console).start()


def available_is_greater(available: str, current: str):
    for available_part, current_part in zip_longest(
        map(int, available.split(".")), map(int, current.split(".")), fillvalue=0
    ):
        if available_part > current_part:
            return True
        if available_part < current_part:
            return False
    return False


def available_is_equal(available: str, current: str):
    for available_part, current_part in zip_longest(
        map(int, available.split(".")), map(int, current.split(".")), fillvalue=0
    ):
        if available_part > current_part:
            return False
        if available_part < current_part:
            return False
    return True


def check_version_on_console():
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.INFO)
    with suppress(Exception):
        warn, message = check_version()
        if warn:
            logging.warning(message)
    logger.setLevel(old_level)


def check_version() -> tuple[bool, str]:
    try:
        out_text = urlopen("https://pypi.org/simple/idtrackerai").read().decode("utf-8")
    except Exception:
        return False, "Could not reach PyPI website to check for updates"

    if not isinstance(out_text, str) or not out_text:
        return False, "Error getting web text"

    # TODO maybe use from html.parser import HTMLParser?
    no_yanked_versions = "\n".join(
        (line for line in out_text.splitlines() if "yanked" not in line)
    )
    versions: list[tuple[str, str]] = re.findall(
        ">idtrackerai-(.+?)(.tar.gz|-py3-none-any.whl)<", no_yanked_versions
    )

    current_version = metadata.version("idtrackerai").split("a")[0]

    current_is_alpha = "a" in metadata.version("idtrackerai")
    for version, _file_extension in versions[::-1]:
        if not version.replace(".", "").isdigit():
            continue  # not a stable version

        if available_is_greater(version, current_version):
            return (
                True,
                (
                    f"A new release of idtracker.ai is available: {current_version} -> "
                    f"{version}\n"
                    'To update, run: "python -m pip install --upgrade idtrackerai"'
                ),
            )
        elif current_is_alpha and available_is_equal(version, current_version):
            return (
                True,
                (
                    "You are running an alpha version of idtracker.ai and the stable"
                    f" version is available: {metadata.version('idtrackerai')} ->"
                    f" {version}\nTo update, run: python -m pip install --upgrade"
                    " idtrackerai"
                ),
            )

    return (
        False,
        (
            "There are currently no updates available.\n"
            f"Current idtrackerai version: {current_version}"
        ),
    )
