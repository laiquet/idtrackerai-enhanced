import logging
from pathlib import Path
from shutil import copy

from idtrackerai import Video
from idtrackerai.animals_detection import detect_animals
from idtrackerai.crossings_detection import CrossingsDetectionAPI
from idtrackerai.fragmentation import fragmentation
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils.py_utils import CheckSegmentationError


def color_log(message: str):
    logging.info("[blue bold]" + message, extra={"markup": True})


class RunIdTrackerAi:
    def __init__(self, GUI_parameters, *args, **kwargs):
        self.user_parameters = GUI_parameters

    def print_final_parameters(self):
        keys_to_print = [
            "session_folder",
            "video_paths",
            "intensity_ths",
            "area_ths",
            "number_of_frames",
            "tracking_intervals",
            "number_of_animals",
            "use_bkg",
            "track_wo_identities",
            "use_ROI",
            "check_segmentation",
            "identity_transfer",
            "knowledge_transfer_folder",
        ]

        params_info = "VIDEO PARAMETERS"

        for key in keys_to_print:
            if key == "video_paths":
                params_info += (
                    f"\n[bold]{key:>20}[/] = {self.video.video_paths[0]}"
                )
                for video_path in self.video.video_paths[1:]:
                    params_info += f"\n{'':>23}{video_path}"
            else:
                params_info += (
                    f"\n[bold]{key:>20}[/] = {getattr(self.video,key)}"
                )
        key = "resolution_reduction"
        params_info += f"\n[bold]{key:>20}[/] = {getattr(self.video,key):.0%}"

        logging.info(params_info, extra={"markup": True})

    def track_video(self):
        global_success = False
        try:
            self.video = Video(**self.user_parameters)  # type: ignore

            self.print_final_parameters()

            self.list_of_blobs = detect_animals(self.video)

            CrossingsDetectionAPI(self.video, self.list_of_blobs)()

            (
                self.list_of_fragments,
                self.list_of_global_fragments,
            ) = fragmentation(
                self.list_of_blobs,
                self.video.number_of_animals,
                self.video.id_images_file_paths,
                self.video.track_wo_identities,
                self.video.fragmentation_time,
            )

            self.tracking()

            global_success = True
            logging.info("Success")
            copy(
                Path("idtrackerai.log"),
                self.video.session_folder / "idtrackerai.log",
            )

        except Exception as e:
            logging.error(
                "An error occurred, saving data before "
                "printing traceback and exiting the program"
            )
            self.save()
            if isinstance(e, CheckSegmentationError):
                # Avoid traceback for check_segmentation
                logging.critical(e, exc_info=False)
            else:
                logging.critical(e, exc_info=True)
                log_file_path = Path("idtrackerai.log").resolve()
                logging.info(
                    "\n\nIf this error persists please let us know by\n"
                    "  - posting on "
                    "https://groups.google.com/g/idtrackerai_users\n"
                    "  - opening an issue at "
                    "https://gitlab.com/polavieja_lab/idtrackerai\n"
                    "  - sending an email to idtrackerai@gmail.com\n"
                    f"Share the log file ({log_file_path}) when "
                    "doing any of the options above"
                )

        return global_success

    def save(self):
        if hasattr(self, "video"):
            self.video.save()
        if hasattr(self, "list_of_blobs"):
            self.list_of_blobs.save(self.video.blobs_path)
        if hasattr(self, "list_of_fragments"):
            self.list_of_fragments.save(self.video.fragments_path)
        if hasattr(self, "list_of_global_fragments"):
            self.list_of_global_fragments.save(
                self.video.global_fragments_path,
                self.list_of_fragments.fragments,
            )

    def tracking(self):
        tracker = TrackerAPI(
            self.video,
            self.list_of_blobs,
            self.list_of_fragments,
            self.list_of_global_fragments,
        )

        if self.video.track_wo_identities:
            color_log("START: TRACKING WITHOUT IDENTITIES")
            tracker.track_wo_identities()
            color_log("FINISH: TRACKING WITHOUT IDENTITIES")
            logging.info(
                "Tracking without identities finished\n"
                "No estimated accuracy computed."
            )
        else:
            if self.video.number_of_animals == 1:
                color_log("START: TRACKING SINGLE ANIMAL")
                tracker.track_single_animal()
                color_log("FINISH: TRACKING SINGLE ANIMAL")

            else:
                color_log("START: TRACKING MULTIPLE ANIMALS")
                tracker.track_multiple_animals()
                color_log("FINISH: TRACKING MULTIPLE ANIMALS")
                self.list_of_fragments.update_id_images_dataset()

            logging.info(
                f"Estimated accuracy: {self.video.estimated_accuracy:.4%}"
            )

            self.video.delete_data()
