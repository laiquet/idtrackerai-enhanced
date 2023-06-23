import logging
import shutil
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

try:
    # PyQt has to be imported before CV2 (importing idtrackerai stuff implies CV2)
    # If not, the QFileDialog.getFileNames() does not load the icons, very weird
    from qtpy.QtWidgets import QApplication
except ImportError:
    logging.error(
        "\n\tRUNNING AN IDTRACKER.AI INSTALLATION WITHOUT ANY QT BINDING.\n\tGUIs are"
        " not available, only tracking directly from the terminal with the `--track`"
        " flag.\n\tRun `pip install pyqt5` or `pip install pyqt6` to build a Qt"
        " binding."
    )

import toml

from idtrackerai.utils import (
    CustomError,
    check_version_on_console_thread,
    initLogger,
    pprint_dict,
    wrap_exceptions,
)

from .arg_parser import parse_args

all_valid_parameters = (
    (Path(__file__).parent / "all_valid_parameters.dat").read_text().splitlines()
)


def load_toml(path: Path, name: str = "") -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} do not exist")
    try:
        toml_dict = {
            key.lower(): value for key, value in toml.load(path.open()).items()
        }

        invalid_keys = [
            key for key in toml_dict.keys() if key not in all_valid_parameters
        ]

        if invalid_keys:
            raise CustomError(
                f"Not recognized parameters while reading {path}: {invalid_keys}"
            )

        for key, value in toml_dict.items():
            if value == "":
                toml_dict[key] = None
        if name:
            logging.info(pprint_dict(toml_dict, name), extra={"markup": True})
        return toml_dict
    except Exception as exc:
        raise CustomError(f"Could not read {path}.\n" + str(exc)) from exc


@wrap_exceptions
def main() -> bool:
    """The command `idtrackerai` runs this function"""
    parameters = {}
    initLogger(check_version=False)

    parameters.update(load_toml((files("idtrackerai") / "constants.toml")))  # type: ignore

    if Path("local_settings.py").is_file():
        logging.warning("Deprecated local_settings format found in ./local_settings.py")

    local_settings_path = Path("local_settings.toml")
    if local_settings_path.is_file():
        local_settings_dict = load_toml(local_settings_path, "Local settings")
        parameters.update(local_settings_dict)

    terminal_args = parse_args(parameters)
    ready_to_track = terminal_args.pop("track")

    if "general_settings" in terminal_args:
        general_settings = load_toml(
            terminal_args.pop("general_settings"), "General settings"
        )
        parameters.update(general_settings)
    else:
        logging.info("No general settings loaded")

    if "session_parameters" in terminal_args:
        session_parameters = load_toml(
            terminal_args.pop("session_parameters"), "Session parameters"
        )
        parameters.update(session_parameters)
    else:
        logging.info("No session parameters loaded")

    if terminal_args:
        logging.info(
            pprint_dict(terminal_args, "Terminal arguments"), extra={"markup": True}
        )
        parameters.update(terminal_args)
    else:
        logging.info("No terminal arguments detected")

    if ready_to_track:
        from .run_idtrackerai import RunIdTrackerAi

        check_version_on_console_thread()

        return RunIdTrackerAi(parameters).track_video()
    run_segmentation_GUI(parameters)
    if parameters.get("run_idtrackerai"):
        from .run_idtrackerai import RunIdTrackerAi

        return RunIdTrackerAi(parameters).track_video()
    return False


def run_segmentation_GUI(params: dict):
    try:
        from idtrackerai_start_app.segmentation_GUI import SegmentationGUI
    except ImportError as exc:
        raise CustomError(
            "\n\tRUNNING AN IDTRACKER.AI INSTALLATION WITHOUT ANY QT BINDING.\n\tGUIs"
            " are not available, only tracking directly from the terminal with the"
            " `--track` flag.\n\tRun `pip install pyqt5` or `pip install pyqt6` to"
            " build a Qt binding."
        ) from exc
    app = QApplication(sys.argv)
    window = SegmentationGUI(params)
    window.show()
    app.exec()


@wrap_exceptions
def general_test():
    from datetime import datetime

    from .run_idtrackerai import RunIdTrackerAi

    COMPRESSED_VIDEO_PATH = Path(str(files("idtrackerai"))) / "data" / "test_B.avi"

    video_path = Path.cwd() / COMPRESSED_VIDEO_PATH.name
    shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)

    initLogger()

    params = load_toml((files("idtrackerai") / "constants.toml"))  # type: ignore
    params.update(
        {
            "session": "test",
            "video_paths": video_path,
            "tracking_intervals": None,
            "intensity_ths": [0, 130],
            "area_ths": [150, 60000],
            "number_of_animals": 8,
            "resolution_reduction": 1.0,
            "check_segmentation": False,
            "ROI_list": None,
            "track_wo_identities": False,
            "use_bkg": False,
            "protocol3_action": "continue",
        }
    )

    start = datetime.now()
    success = RunIdTrackerAi(params).track_video()
    if success:
        logging.info(
            "[green]Test passed successfully in %s with version %s",
            str(datetime.now() - start).split(".")[0],
            version("idtrackerai"),
            extra={"markup": True},
        )


if __name__ == "__main__":
    main()
