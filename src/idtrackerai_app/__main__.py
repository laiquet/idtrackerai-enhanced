import sys, os
from threading import local
from PyQt6.QtWidgets import QApplication
import logging
from rich.logging import RichHandler
from rich.console import Console
import importlib.metadata
import argparse
import shutil
from idtrackerai_app.run_idtrackerai import RunIdTrackerAi
import pydoc

sys.path.append(os.getcwd())


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
        datefmt="%b %d %H:%M:%S",
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

    logger = logging.getLogger()
    logging.getLogger("PyQt6").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logger.info("Welcome to idtracker.ai")
    logger.debug(
        f"Running idtracker {importlib.metadata.version('idtrackerai')}"
        f" on Python {sys.version.split(' ')[0]}"
    )


def start():
    init_logger()
    logger = logging.getLogger()
    from confapp import conf

    try:
        import local_settings

        local_settings.SETTINGS_PRIORITY = 10
        conf += local_settings
        logger.info("Local settings file found with:")
        printing = False
        for line in pydoc.plain(pydoc.render_doc(local_settings)).split("\n"):
            if line == "":
                printing = False
            if printing:
                logger.info(line)
            if line == "DATA":
                printing = True

    except ImportError:
        logger.info("Local settings file not found")

    # import idtrackerai
    # idtrackerai.constants.SETTINGS_PRIORITY = 2
    # conf += idtrackerai.constants

    # with open("/home/jordi/idtrackerai/no_name_session.json", "r") as f:
    #     GUI_parameters = json.load(f)
    # print(GUI_parameters)
    GUI_parameters = {}

    if True:
        from .GUI_main import Window

        app = QApplication(sys.argv)
        window = Window(GUI_parameters)
        window.show()
        app.exec()

    if GUI_parameters.get("run_idtrackerai", False):
        RunIdTrackerAi(GUI_parameters).track_video()


def general_test():
    from idtrackerai.constants import (
        IDTRACKERAI_FOLDER,
        COMPRESSED_VIDEO_PATH,
    )

    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
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
        print("Copying test video file to: {args.output_folder}")
        _, video_name = os.path.split(COMPRESSED_VIDEO_PATH)
        video_path = os.path.join(args.output_folder, video_name)
        shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)
    else:
        video_path = COMPRESSED_VIDEO_PATH

    json_content = {
        "open-multiple-files": False,
        "session": "test",
        "video_paths": video_path,
        "range": None,
        "intensity": [0, 155],
        "area": [150, 60000],
        "number_of_animals": 8,
        "resreduct": 1.0,
        "chcksegm": False,
        "roi": None,
        "no_ids": args.no_identities,
        "bgsub": False,
    }

    RunIdTrackerAi(json_content).track_video()


# Execute the application
if __name__ == "__main__":
    start()
