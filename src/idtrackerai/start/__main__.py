import logging
import shutil
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

from idtrackerai import Session
from idtrackerai.utils import (
    IdtrackeraiError,
    conf,
    load_toml,
    manage_exception,
    pprint_dict,
    wrap_entrypoint,
)

from .arg_parser import parse_args


def gather_input_parameters() -> tuple[bool, dict[str, Any]]:
    parameters = {}
    if Path("local_settings.py").is_file():
        logging.warning("Deprecated local_settings format found in ./local_settings.py")

    local_settings_path = Path("local_settings.toml")
    if local_settings_path.is_file():
        parameters = load_toml(local_settings_path)

    terminal_args = parse_args()
    ready_to_track = terminal_args.pop("track")

    if "general_settings" in terminal_args:
        parameters.update(load_toml(terminal_args.pop("general_settings")))
        logging.warning(
            "The terminal argument --settings is deprecated, please use --load with"
            " multiple files instead."
        )

    if "parameters" in terminal_args:
        for parameter_file in terminal_args.pop("parameters"):
            parameters.update(load_toml(parameter_file))
    else:
        logging.info("No parameter files detected")

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

    session = Session()
    non_recognized_params_1 = conf.set_parameters(**user_parameters)
    non_recognized_params_2 = session.set_parameters(**user_parameters)

    non_recognized_params = non_recognized_params_1 & non_recognized_params_2

    if non_recognized_params:
        raise IdtrackeraiError(f"Not recognized parameters: {non_recognized_params}")

    if not ready_to_track:
        ready_to_track = run_segmentation_GUI(session)
        if not ready_to_track:
            return False

    from idtrackerai.base.run import RunIdTrackerAi

    return RunIdTrackerAi(session).track_video()


def run_segmentation_GUI(session: Session | None) -> bool:
    try:
        from qtpy.QtWidgets import QApplication

        from idtrackerai.segmentation_app import SegmentationGUI
    except ImportError as exc:
        raise IdtrackeraiError(
            "\n\tRUNNING AN IDTRACKER.AI INSTALLATION WITHOUT ANY QT BINDING.\n\tGUIs"
            " are not available, only tracking directly from the terminal with the"
            " `--track` flag.\n\tRun `pip install pyqt5` or `pip install pyqt6` to"
            f" build a Qt binding. Original Exception: {exc}"
        ) from exc

    # this catches exceptions when raised inside Qt
    def excepthook(exc_type, exc_value, exc_tb):
        assert QApplication  # Pylance is happier with this
        QApplication.quit()
        manage_exception(exc_value)

    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    signal = {"run_idtrackerai": False}
    window = SegmentationGUI(session, signal)
    window.show()
    app.exec()
    return signal["run_idtrackerai"] is True


@wrap_entrypoint
def general_test():
    from datetime import datetime

    from idtrackerai.base.run import RunIdTrackerAi

    COMPRESSED_VIDEO_PATH = Path(str(files("idtrackerai"))) / "data" / "test_B.avi"

    video_path = Path.cwd() / COMPRESSED_VIDEO_PATH.name
    shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)

    session = Session()
    session.set_parameters(
        name="test",
        video_paths=video_path,
        tracking_intervals=None,
        intensity_ths=[0, 130],
        area_ths=[150, float("inf")],
        number_of_animals=8,
        resolution_reduction=1.0,
        check_segmentation=False,
        ROI_list=None,
        track_wo_identities=False,
        use_bkg=False,
        protocol3_action="continue",
    )

    start = datetime.now()
    success = RunIdTrackerAi(session).track_video()
    if success:
        logging.info(
            "[green]Test passed successfully in %s with version %s",
            str(datetime.now() - start).split(".")[0],
            version("idtrackerai"),
            extra={"markup": True},
        )


if __name__ == "__main__":
    main()
