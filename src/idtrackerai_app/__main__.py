import sys
import os
import logging
from rich.logging import RichHandler
from rich.console import Console
from importlib import metadata
from argparse import ArgumentParser
import shutil
from idtrackerai_app.run_idtrackerai import RunIdTrackerAi
import pydoc
from pathlib import Path
import json
import ast


def init_logger():
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
        level=0,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RichHandler(console=Console(width=size)),
            RichHandler(
                console=Console(
                    file=open("idtrackerai-app.log", "w"),
                    width=logger_width_when_no_terminal,
                ),
            ),
        ],
    )

    logging.getLogger("PyQt6").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.info("Welcome to idTracker.ai")
    logging.debug(
        f"Running idTracker.ai {metadata.version('idtrackerai')}"
        f" on Python {sys.version.split(' ')[0]}"
    )


def to_bool(value):
    valid = {
        "true": True,
        "t": True,
        "1": True,
        "false": False,
        "f": False,
        "0": False,
    }

    if isinstance(value, bool):
        return value

    if not isinstance(value, str):
        raise ValueError("invalid literal for boolean. Not a string.")

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError(f'invalid literal for boolean: "{value}"')


def start(input_parameters={}, test=False):
    init_logger()
    from confapp import conf

    try:
        sys.path.append(".")
        import local_settings

        to_print = "Local settings file found with:\n"
        printing = False
        for line in pydoc.plain(pydoc.render_doc(local_settings)).splitlines():
            if line == "":
                printing = False
            if printing:
                to_print += line + "\n"
            if line == "DATA":
                printing = True
        logging.info(to_print)

        local_settings.SETTINGS_PRIORITY = 10
        conf += local_settings

    except ImportError:
        logging.info("Local settings file not found")

    import idtrackerai

    idtrackerai.constants.SETTINGS_PRIORITY = 2
    conf += idtrackerai.constants

    defaults = {
        "tracking_intervals": "all",
        "resolution_reduction": 1,
        "check_segmentation": False,
        "ROI_list": None,
        "use_bkg": False,
        "setup_points": None,
        "track_wo_identities": False,
    }

    user_parameters = {}
    user_parameters.update(defaults)
    user_parameters.update(input_parameters)

    to_print = "Default parameters:\n"
    for key, item in user_parameters.items():
        to_print += f"[bold]{key:>{23}}[/] = {item}\n"
    logging.info(to_print, extra={"markup": True})

    keys = (
        (
            "tracking_intervals",
            "Tracking intervals (in frames)",
            ast.literal_eval,
        ),
        ("intensity_ths", "Pixel's intensity thresholds", ast.literal_eval),
        ("area_ths", "Blob's areas thresholds", ast.literal_eval),
        (
            "number_of_animals",
            "Number of different animals that appear in the video",
            int,
        ),
        ("resolution_reduction", "Video resolution reduction ratio", float),
        (
            "check_segmentation",
            "Check all frames have less or equal number of blobs than animals",
            to_bool,
        ),
        ("ROI_list", "List of polygons defining the Region Of Interest", str),
        (
            "use_bkg",
            "Compute and extract background to improve blob identification",
            to_bool,
        ),
        (
            "setup_points",
            "User defined points in the video frame, no effect on tracking",
            str,
        ),
        ("video_paths", "List of paths to the video files to track", str),
        ("session", "Name of the session", str),
        (
            "track_wo_identities",
            "Track the video ignoring identities (without AI)",
            to_bool,
        ),
    )

    parser = ArgumentParser(prog="idTracker.ai")

    parser.add_argument(
        "--load",
        help=".JSON file to load",
        type=os.path.abspath,
        dest="user_params",
    )

    parser.add_argument(
        "--track",
        help="Track the video without launching the GUI. Default False",
        action="store_true",
    )

    for key, description, dtype in keys:
        if key in ("video_paths", "tracking_intervals"):
            nargs = "+"
        else:
            nargs = None
        parser.add_argument(
            "--" + key,
            default=-1,
            help=description,
            type=dtype,
            nargs=nargs,
            metavar=key.title(),
        )

    args = parser.parse_args()

    try:
        if args.user_params:
            with open(args.user_params) as f:
                json_file = json.load(f)
                to_print = f"Loading .JSON input file {args.user_params}\n"
                for key, item in json_file.items():
                    to_print += f"[bold]{key:>{23}}[/] = {item}\n"
                logging.info(to_print, extra={"markup": True})
                user_parameters.update(json_file)
        else:
            logging.info("No .JSON input file to load")

    except Exception as e:
        logging.error(
            f"Error while reading '{args.user_params}':\n"
            f"\t{e}\nIgnoring '--load' terminal argument"
        )

    to_print = "Reading terminal argument:\n"
    any_loaded = True
    for key, description, dtype in keys:
        arg = getattr(args, key)
        if arg != -1:
            any_loaded = False
            to_print += f"[bold]{key:>{23}}[/] = {arg}\n"
            user_parameters[key] = arg
    if any_loaded:
        logging.info("No terminal arguments detected")
    else:
        logging.info(to_print, extra={"markup": True})

    if args.track or test:
        success = RunIdTrackerAi(user_parameters).track_video()
        return success
    else:
        run_app(user_parameters)
        if user_parameters.get("run_idtrackerai", False):
            success = RunIdTrackerAi(user_parameters).track_video()
            return success


def run_app(params: dict):
    from PyQt6.QtWidgets import QApplication
    from idtrackerai_app import Window
    from PyQt6.QtWidgets import QStyleFactory
    from .themes import create_palette

    app = QApplication(sys.argv)

    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
        app.setPalette(create_palette(style="custom"))
    else:
        logging.info(
            "'Fusion' style not found on current PyQt6"
            "installation, ignoring custom dark theme"
        )

    window = Window(params)
    window.show()
    app.exec()


def general_test():
    init_logger()
    import idtrackerai
    from idtrackerai.constants import COMPRESSED_VIDEO_PATH

    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_folder",
        type=Path,
        help="Path to the folder where the video will be stored",
    )
    parser.add_argument(
        "-n",
        "--no_identities",
        action="store_true",
        help="Flag to track without identities",
    )
    args = parser.parse_args()

    if args.output_folder:
        logging.info(f"Copying test video file to: {args.output_folder}")
        video_path = args.output_folder / COMPRESSED_VIDEO_PATH.name
        shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)
    else:
        video_path = COMPRESSED_VIDEO_PATH

    json_content = {
        "session": "test",
        "video_paths": video_path,
        "tracking_intervals": None,
        "intensity_ths": [0, 155],
        "area_ths": [150, 60000],
        "number_of_animals": 8,
        "resolution_reduction": 1.0,
        "check_segmentation": False,
        "ROI_list": None,
        "track_wo_identities": args.no_identities,
        "use_bkg": False,
    }

    from confapp import conf

    idtrackerai.constants.SETTINGS_PRIORITY = 2
    conf += idtrackerai.constants

    return RunIdTrackerAi(json_content).track_video()


# Execute the application
if __name__ == "__main__":
    start()
