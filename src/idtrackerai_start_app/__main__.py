import logging
import shutil
import sys
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path

import toml

from idtrackerai.utils import conf, pprint_dict, initLogger

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
            logging.error(
                f"Not recognized parameters while reading {path}: {invalid_keys}"
            )
            exit()

        for key, value in toml_dict.items():
            if value == "":
                toml_dict[key] = None
        if name:
            logging.info(pprint_dict(toml_dict, name), extra={"markup": True})
        return toml_dict
    except Exception:
        logging.error(f"Could not read {path}, bad format")
        exit()


def main() -> bool:
    """The command `idtrackerai` runs this function"""
    parameters = {}
    initLogger()

    constants = load_toml((files("idtrackerai") / "constants.toml"))  # type: ignore
    parameters.update(constants)

    if Path("local_settings.py").is_file():
        logging.warning("Deprecated local_settings format found in ./local_settings.py")

    local_settings_path = Path("local_settings.toml")
    if local_settings_path.is_file():
        local_settings_dict = load_toml(local_settings_path, "Local settings")
        parameters.update(local_settings_dict)

    conf.set_dict(constants)  # this enables defaults in terminal argument parser
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

    if ready_to_track:
        from .run_idtrackerai import RunIdTrackerAi

        return RunIdTrackerAi(parameters).track_video()
    else:
        run_segmentation_GUI(parameters)
        if parameters.get("run_idtrackerai", False):
            from .run_idtrackerai import RunIdTrackerAi

            return RunIdTrackerAi(parameters).track_video()
        return False


def run_segmentation_GUI(params: dict):
    from PyQt6.QtWidgets import QApplication, QStyleFactory

    from idtrackerai_start_app.segmentation_GUI import SegmentationGUI

    app = QApplication(sys.argv)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    window = SegmentationGUI(params)
    window.show()
    app.exec()


def general_test():
    COMPRESSED_VIDEO_PATH = Path(str(files("idtrackerai"))) / "data" / "example_B.avi"

    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="Path to the folder where the video will be stored",
    )
    args = parser.parse_args()

    if args.output_dir:
        video_path = args.output_dir / COMPRESSED_VIDEO_PATH.name
        shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)
    else:
        video_path = COMPRESSED_VIDEO_PATH

    initLogger(testing=True)

    params = load_toml((files("idtrackerai") / "constants.toml"))  # type: ignore
    params.update(
        {
            "session": "test",
            "video_paths": video_path,
            "tracking_intervals": None,
            "intensity_ths": [0, 155],
            "area_ths": [150, 60000],
            "number_of_animals": 8,
            "resolution_reduction": 1.0,
            "check_segmentation": False,
            "ROI_list": None,
            "track_wo_identities": False,
            "use_bkg": False,
        }
    )
    from .run_idtrackerai import RunIdTrackerAi

    RunIdTrackerAi(params).track_video()

    if not args.output_dir:
        shutil.rmtree(COMPRESSED_VIDEO_PATH.parent / "session_test", ignore_errors=True)


if __name__ == "__main__":
    main()
