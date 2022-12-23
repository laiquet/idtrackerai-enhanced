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


def check_version():
    return
    try:
        response = requests.get(
            "https://pypi.org/pypi/idtrackerai/json", timeout=1
        )
    except requests.exceptions.RequestException:
        return
    if response.status_code != 200:
        return

    data = response.json()

    versions: list[str] = list(data["releases"].keys())

    for available_version in versions[::-1]:
        if available_version.replace(".", "").isdigit():
            break

    upload_date = None
    for file_info in data["releases"][available_version]:
        if file_info["upload_time"]:
            upload_date = file_info["upload_time"][:10]
            break

    current_version = idtrackerai.__version__
    if available_is_greater(available_version, current_version):
        logging.warning(
            f"The new idtracker.ai {available_version} (released in "
            f"{upload_date}) is now available on PyPI.\nSince "
            f"you are running idtracker.ai {current_version}, we encourage "
            "you to upgrade by running 'pip install --upgrade idtrackerai'"
        )
