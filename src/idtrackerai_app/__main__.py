import sys
import os
from PyQt6.QtWidgets import QApplication
import logging
from rich.logging import RichHandler
from rich.console import Console
import importlib.metadata
from argparse import ArgumentParser
import shutil
from idtrackerai_app.run_idtrackerai import RunIdTrackerAi
import pydoc
from pathlib import Path
import json


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
            RichHandler(console=Console(width=size), markup=False),
            RichHandler(
                console=Console(
                    file=open("idtrackerai-app.log", "w"),
                    width=logger_width_when_no_terminal,
                    markup=False,
                ),
            ),
        ],
    )

    logging.getLogger("PyQt6").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.info("Welcome to idtracker.ai")
    logging.debug(
        f"Running idtracker {importlib.metadata.version('idtrackerai')}"
        f" on Python {sys.version.split(' ')[0]}"
    )


def start():
    init_logger()
    from confapp import conf

    try:
        import local_settings

        local_settings.SETTINGS_PRIORITY = 10
        conf += local_settings
        logging.info("Local settings file found with:")
        printing = False
        for line in pydoc.plain(pydoc.render_doc(local_settings)).splitlines():
            if line == "":
                printing = False
            if printing:
                logging.info(line)
            if line == "DATA":
                printing = True

    except ImportError:
        logging.info("Local settings file not found")

    import idtrackerai

    idtrackerai.constants.SETTINGS_PRIORITY = 2
    conf += idtrackerai.constants

    parser = ArgumentParser(prog="idTracker.ai")
    parser.add_argument(
        "--load",
        help=".JSON file to load",
        type=Path,
        dest="user_params",
    )
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()

    try:
        if args.user_params:
            with open(args.user_params) as f:
                user_parameters = json.load(f)
        else:
            user_parameters = {}
    except Exception as e:
        error_msg = (
            f"Error while reading '{args.user_params}':\n"
            f"\t{e}\n"
            "Ignoring '--load' terminal argument"
        )
        logging.error(error_msg)
        user_parameters = {}

    if args.track:
        success = RunIdTrackerAi(user_parameters).track_video()
        return success
    else:
        run_app(user_parameters)
        if user_parameters.get("run_idtrackerai", False):
            success = RunIdTrackerAi(user_parameters).track_video()
            return success


background = "#202124"
mid_background = "#2C2D2F"
light_bkg = "#3F4042"
blue = "#8AB4F7"
almost_white = "#FDFDFD"
placeholder_color = "#B0B0B0"
red = "#FF0000"


def run_app(params):
    from idtrackerai_app import Window

    # import qdarktheme
    from PyQt6.QtGui import QPalette, QColor
    from matplotlib.pyplot import rcParams

    app = QApplication(sys.argv)

    rcParams.update(
        {
            "ytick.color": almost_white,
            "xtick.color": almost_white,
            "axes.labelcolor": almost_white,
            "axes.edgecolor": almost_white,
            "text.color": almost_white,
            "figure.facecolor": background,
        }
    )
    palette = app.palette()
    app.setStyle("Windows")

    palette.setColor(QPalette.Window, QColor(background))
    palette.setColor(QPalette.WindowText, QColor(almost_white))
    palette.setColor(QPalette.Base, QColor(light_bkg))
    palette.setColor(QPalette.AlternateBase, QColor(mid_background))
    palette.setColor(QPalette.Text, QColor(almost_white))
    palette.setColor(QPalette.Button, QColor(mid_background))
    palette.setColor(QPalette.ButtonText, QColor(blue))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(2, 2, 2))
    palette.setColor(QPalette.PlaceholderText, QColor(placeholder_color))
    palette.setColor(QPalette.BrightText, QColor(253, 2, 2))

    # palette.setColor(QPalette.Light, QColor(blue))
    # palette.setColor(QPalette.Midlight, QColor(blue))
    # palette.setColor(QPalette.Dark, QColor(blue))
    # palette.setColor(QPalette.Shadow, QColor(blue))

    # palette.setColor(QPalette.Mid, QColor(255, 0, 0))
    # palette.setColor(QPalette.Link, QColor(255, 0, 0))
    # palette.setColor(QPalette.LinkVisited, QColor(255, 0, 0))
    # palette.setColor(QPalette.ToolTipBase, QColor(255, 0, 0))
    # palette.setColor(QPalette.ToolTipText, QColor(255, 0, 0))
    # palette.setColor(QPalette.NoRole, QColor(255, 0, 0))

    # palette.setColor(QPalette.Button, QColor(red))

    app.setPalette(palette)
    window = Window(params)

    # app.setStyleSheet(qdarktheme.load_stylesheet())

    window.show()
    app.exec()


def general_test():
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
        "no_ids": args.no_identities,
        "use_bkg": False,
    }

    init_logger()
    from confapp import conf

    idtrackerai.constants.SETTINGS_PRIORITY = 2
    conf += idtrackerai.constants

    return RunIdTrackerAi(json_content).track_video()


# Execute the application
if __name__ == "__main__":
    start()
