import logging
import shutil
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

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

from idtrackerai import Video
from idtrackerai.utils import CustomError, conf, pprint_dict, wrap_entrypoint

from .arg_parser import parse_args


def load_toml(path: Path, name: str = "") -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} do not exist")
    try:
        toml_dict = {
            key.lower(): value for key, value in toml.load(path.open()).items()
        }

        for key, value in toml_dict.items():
            if value == "":
                toml_dict[key] = None
        if name:
            logging.info(pprint_dict(toml_dict, name), extra={"markup": True})
        return toml_dict
    except Exception as exc:
        raise CustomError(f"Could not read {path}.\n" + str(exc)) from exc


def gather_input_parameters() -> tuple[bool, dict[str, Any]]:
    parameters = {}
    if Path("local_settings.py").is_file():
        logging.warning("Deprecated local_settings format found in ./local_settings.py")

    local_settings_path = Path("local_settings.toml")
    if local_settings_path.is_file():
        parameters = load_toml(local_settings_path, "Local settings")

    terminal_args = parse_args()
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
    return ready_to_track, parameters


@wrap_entrypoint
def main() -> bool:
    """The command `idtrackerai` runs this function"""
    ready_to_track, user_parameters = gather_input_parameters()

    video = Video()
    non_recognized_params_1 = conf.set_parameters(**user_parameters)
    non_recognized_params_2 = video.set_parameters(**user_parameters)

    non_recognized_params = non_recognized_params_1 & non_recognized_params_2

    if non_recognized_params:
        raise CustomError(f"Not recognized parameters: {non_recognized_params}")

    if not ready_to_track:
        ready_to_track = run_segmentation_GUI(video)
        if not ready_to_track:
            return False

    from .run_idtrackerai import RunIdTrackerAi

    return RunIdTrackerAi(video).track_video()


def run_segmentation_GUI(video: Video | None) -> bool:
    try:
        from idtrackerai_start_app.segmentation_GUI import SegmentationGUI
    except ImportError as exc:
        raise CustomError(
            "\n\tRUNNING AN IDTRACKER.AI INSTALLATION WITHOUT ANY QT BINDING.\n\tGUIs"
            " are not available, only tracking directly from the terminal with the"
            " `--track` flag.\n\tRun `pip install pyqt5` or `pip install pyqt6` to"
            " build a Qt binding."
        ) from exc
    assert QApplication  # Pylance is happier with this
    app = QApplication(sys.argv)
    signal = {"run_idtrackerai": False}
    window = SegmentationGUI(video, signal)
    window.show()
    app.exec()
    return signal["run_idtrackerai"] is True


@wrap_entrypoint
def general_test():
    from datetime import datetime

    from .run_idtrackerai import RunIdTrackerAi

    COMPRESSED_VIDEO_PATH = Path(str(files("idtrackerai"))) / "data" / "test_B.avi"

    video_path = Path.cwd() / COMPRESSED_VIDEO_PATH.name
    shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)

    video = Video()
    video.set_parameters(
        session="test",
        video_paths=video_path,
        tracking_intervals=None,
        intensity_ths=[0, 130],
        area_ths=[150, 60000],
        number_of_animals=8,
        resolution_reduction=1.0,
        check_segmentation=False,
        ROI_list=None,
        track_wo_identities=False,
        use_bkg=False,
        protocol3_action="continue",
    )

    start = datetime.now()
    success = RunIdTrackerAi(video).track_video()
    if success:
        logging.info(
            "[green]Test passed successfully in %s with version %s",
            str(datetime.now() - start).split(".")[0],
            version("idtrackerai"),
            extra={"markup": True},
        )


if __name__ == "__main__":
    main()
