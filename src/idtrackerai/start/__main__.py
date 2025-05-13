import logging
import multiprocessing
import shutil
import sys
from argparse import ArgumentError
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

from idtrackerai import IdtrackeraiError, Session, conf
from idtrackerai.utils import (
    LOGGING_QUEUE,
    load_toml,
    manage_exception,
    pprint_dict,
    setup_logging_queue,
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

    try:
        terminal_args = parse_args()
    except ArgumentError as exc:
        raise IdtrackeraiError() from exc

    ready_to_track = terminal_args.pop("track")

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

    if "add_time_column_to_csv" in non_recognized_params:
        raise IdtrackeraiError(
            'The parameter "add_time_column_to_csv" has been removed and set to always True'
        )
    if "convert_trajectories_to_csv_and_json" in non_recognized_params:
        raise IdtrackeraiError(
            'The parameter "CONVERT_TRAJECTORIES_TO_CSV_AND_JSON" has been replaced by '
            '"TRAJECTORIES_FORMATS"\nCheck the documentation https://idtracker.ai/latest/user_guide/usage.html#output'
        )

    p3_deprecated = {
        "protocol3_action",
        "threshold_acceptable_accumulation",
        "maximum_number_of_parachute_accumulations",
        "max_ratio_of_pretrained_images",
    }
    if p3_deprecated & non_recognized_params:
        raise IdtrackeraiError(
            "The following parameters have been removed from idtracker.ai "
            f"because Protocol 3 has been replaced by Contrastive Protocol:\n{p3_deprecated}"
        )

    if non_recognized_params:
        raise IdtrackeraiError(f"Not recognized parameters: {non_recognized_params}")

    if not ready_to_track:
        ready_to_track = run_segmentation_GUI(session)
        if not ready_to_track:
            return False

    from idtrackerai.base.run import RunIdTrackerAi

    return RunIdTrackerAi(session).track_video()


def run_segmentation_GUI(session: Session | None = None) -> bool:
    """Run the segmentation GUI in a separate process to catch segmentation faults"""
    # https://stackoverflow.com/a/10415215
    communication_queue = multiprocessing.Queue()

    if sys.platform.startswith("win") or sys.platform == "darwin":
        communication_dict = run_gui_in_main_process(session)
    else:
        # On Linux, we run the GUI in a separate process to catch libxcb errors
        p = multiprocessing.Process(
            target=run_gui_in_child_process,
            args=(session, communication_queue, LOGGING_QUEUE),
        )
        p.start()
        p.join()

        if p.exitcode == -6 and sys.platform == "linux":
            raise RuntimeError(
                f"QApplication crashed with exit code {p.exitcode}. This is a known issue with the "
                "Qt library on Linux. Try to run 'sudo apt install libxcb-cursor0'."
            )
        elif p.exitcode != 0:
            raise RuntimeError(f"QApplication crashed with exit code {p.exitcode}")

        communication_dict = communication_queue.get()

    run_idtrackerai = communication_dict.get("run_idtrackerai", False)
    if session is not None:
        session.__dict__.update(communication_dict)
    return run_idtrackerai


def run_gui_in_child_process(
    session: Session | None,
    communication_queue: multiprocessing.Queue,
    logging_queue: multiprocessing.Queue,
) -> None:
    if logging_queue:
        setup_logging_queue(logging_queue)

    try:
        from qtpy.QtWidgets import QApplication

        from idtrackerai.segmentation_app import SegmentationGUI
    except ImportError as exc:
        raise IdtrackeraiError(
            "\n\tRUNNING AN IDTRACKER.AI INSTALLATION WITHOUT ANY QT BINDING.\n\tGUIs"
            " are not available, only tracking directly from the terminal with the"
            " `--track` flag.\n\tRun `pip install pyqt6` or `pip install pyqt5` to"
            " build a Qt binding"
        ) from exc

    # this catches exceptions when raised inside Qt
    def excepthook(exc_type, exc_value, exc_tb):
        assert QApplication  # Pylance is happier with this
        QApplication.quit()
        manage_exception(exc_value)

    sys.excepthook = excepthook
    app = QApplication(sys.argv)

    window = SegmentationGUI(session)
    window.show()
    app.exec()

    # communicate to main process the new session parameters and if the user wants to track
    communication_queue.put(
        (session.__dict__ if session is not None else {})
        | {"run_idtrackerai": window.run_idtrackerai_after_closing}
    )


def run_gui_in_main_process(session: Session | None) -> dict[str, Any]:
    try:
        from qtpy.QtWidgets import QApplication

        from idtrackerai.segmentation_app import SegmentationGUI
    except ImportError as exc:
        raise IdtrackeraiError(
            "\n\tRUNNING AN IDTRACKER.AI INSTALLATION WITHOUT ANY QT BINDING.\n\tGUIs"
            " are not available, only tracking directly from the terminal with the"
            " `--track` flag.\n\tRun `pip install pyqt6` or `pip install pyqt5` to"
            " build a Qt binding"
        ) from exc

    # this catches exceptions when raised inside Qt
    def excepthook(exc_type, exc_value, exc_tb):
        assert QApplication  # Pylance is happier with this
        QApplication.quit()
        manage_exception(exc_value)

    sys.excepthook = excepthook
    app = QApplication(sys.argv)

    window = SegmentationGUI(session)
    window.show()
    app.exec()

    return (session.__dict__ if session is not None else {}) | {
        "run_idtrackerai": window.run_idtrackerai_after_closing
    }


@wrap_entrypoint
def general_test():
    from datetime import datetime

    logging.info("Starting general idtrackerai test")

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
        check_segmentation=False,
        roi_list=None,
        track_wo_identities=False,
        use_bkg=False,
    )

    _ready_to_track, user_parameters = gather_input_parameters()
    non_recognized_params_1 = conf.set_parameters(**user_parameters)
    non_recognized_params_2 = session.set_parameters(**user_parameters)
    non_recognized_params = non_recognized_params_1 & non_recognized_params_2
    if non_recognized_params:
        raise IdtrackeraiError(f"Not recognized parameters: {non_recognized_params}")
    from idtrackerai.base.run import RunIdTrackerAi

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
