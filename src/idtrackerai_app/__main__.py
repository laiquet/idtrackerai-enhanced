import ast
import logging
import os
import shutil
import sys
from argparse import ArgumentParser
from importlib import metadata
from importlib.resources import files
from pathlib import Path
from platform import platform

import toml
from rich.console import Console
from rich.logging import RichHandler

from .check_PyPI_version import check_version


def init_logger(testing=False):
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
                    file=open("idtrackerai.log", "w"),
                    width=logger_width_when_no_terminal,
                )
            ),
        ],
    )

    logging.getLogger("PyQt6").setLevel(logging.INFO)
    logging.info("Welcome to idTracker.ai")
    logging.debug(
        f"Running idTracker.ai {metadata.version('idtrackerai')}"
        f" on Python {sys.version.split(' ')[0]}\n"
        f"Platform: {platform(True)}"
    )
    check_version()


def to_bool(value):
    valid = {"true": True, "t": True, "1": True, "false": False, "f": False, "0": False}

    if isinstance(value, bool):
        return value

    if not isinstance(value, str):
        raise ValueError("invalid literal for boolean. Not a string.")

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError(f'invalid literal for boolean: "{value}"')


def main(input_parameters={}, test=False) -> bool:
    init_logger(testing=test)
    from idtrackerai.utils import conf

    conf.reset_all()

    if not test:

        if Path("local_settings.py").is_file():
            logging.warning(
                "Deprecated local_settings format found in ./local_settings.py"
            )

        loca_settings_path = Path("local_settings.toml")
        if Path("local_settings.toml").is_file():
            local_settings_dict = toml.load(loca_settings_path.open())
            conf.set_dict(local_settings_dict, "local_settings", 2)
        else:
            logging.info(f"Local settings file not found in {loca_settings_path}")

    constants = toml.load((files("idtrackerai") / "constants.toml").open())

    for key, value in constants.items():
        if value == "":
            constants[key] = None
        if key in os.environ:
            constants[key] = os.environ[key]

    conf.set_dict(constants, "constants", priority=1, verbose=False)

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

    from idtrackerai_app import RunIdTrackerAi

    if test:
        conf.set_dict(user_parameters, "user_parameters", 3)
        return RunIdTrackerAi(user_parameters).track_video()

    keys = (
        ("tracking_intervals", "Tracking intervals (in frames)", ast.literal_eval),
        ("intensity_ths", "Pixel's intensity thresholds", ast.literal_eval),
        ("area_ths", "Blob's areas thresholds", ast.literal_eval),
        (
            "number_of_animals",
            "Number of different animals that appear in the video",
            int,
        ),
        ("output_dir", "Output directory, Default is video paths directory", Path),
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
        "--load", help=".TOML file to load", type=Path, dest="user_params"
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
            nargs=nargs, # type: ignore
            metavar=key.title(),
        )

    args = parser.parse_args()

    try:
        if args.user_params:
            toml_file = toml.load(args.user_params.open())
            to_print = f"Loading .TOML input file {args.user_params}\n"
            for key, item in toml_file.items():
                to_print += f"[bold]{key:>{23}}[/] = {item}\n"
            logging.info(to_print, extra={"markup": True})
            user_parameters.update(toml_file)
        else:
            logging.info("No .TOML input file to load")

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

    conf.set_dict(user_parameters, "user_parameters")

    if args.track:
        success = RunIdTrackerAi(user_parameters).track_video()
        return success
    else:
        run_app(user_parameters)
        if user_parameters.get("run_idtrackerai", False):
            success = RunIdTrackerAi(user_parameters).track_video()
            return success
    return False


def run_app(params: dict):
    from idtrackerai_app import SegmentationGUI
    from PyQt6.QtWidgets import QApplication, QStyleFactory

    app = QApplication(sys.argv)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    window = SegmentationGUI(params)
    window.show()
    app.exec()


def general_test():
    COMPRESSED_VIDEO_PATH = (
        Path(str(files("idtrackerai")))
        / "data"
        / "example_video_compressed"
        / "conflict3and4_20120316T155032_14_compressed.avi"
    )

    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
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

    if args.output_dir:
        video_path = args.output_dir / COMPRESSED_VIDEO_PATH.name
        shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)
    else:
        video_path = COMPRESSED_VIDEO_PATH

    params = {
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

    main(params, test=True)


# Execute the application
if __name__ == "__main__":
    main()
