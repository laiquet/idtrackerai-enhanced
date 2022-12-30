import logging
from pathlib import Path
from shutil import copy

from idtrackerai import Video
from idtrackerai.animals_detection import animals_detection_API
from idtrackerai.crossings_detection import crossings_detection_API
from idtrackerai.fragmentation import fragmentation_API
from idtrackerai.postprocess import trajectories_API
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils import CheckSegmentationError


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

    def track_video(self) -> bool:
        try:
            self.video = Video(**self.user_parameters)  # type: ignore
            self.print_final_parameters()

            self.save()

            self.list_of_blobs = animals_detection_API(self.video)

            self.save()

            crossings_detection_API(self.video, self.list_of_blobs)

            self.save()

            (
                self.list_of_fragments,
                self.list_of_global_fragments,
            ) = fragmentation_API(self.video, self.list_of_blobs)
            self.save()

            tracker = TrackerAPI(
                self.video,
                self.list_of_blobs,
                self.list_of_fragments,
                self.list_of_global_fragments,
            )

            if not self.video.track_wo_identities:
                if self.video.single_animal:
                    tracker.track_single_animal()
                else:
                    print(self.list_of_global_fragments.single_global_fragment)
                    if self.list_of_global_fragments.single_global_fragment:
                        tracker.track_single_global_fragment_video()
                    else:
                        self.list_of_fragments = (
                            tracker.track_with_identities()
                        )
                        self.list_of_fragments.update_id_images_dataset()

            self.save()

            trajectories_API(
                self.video,
                self.list_of_blobs,
                self.list_of_global_fragments.single_global_fragment,
                self.list_of_fragments,
            )

            if self.video.track_wo_identities:
                logging.info(
                    "Tracking without identities finished\n"
                    "No estimated accuracy computed."
                )
            else:
                logging.info(
                    f"Estimated accuracy: {self.video.estimated_accuracy:.4%}"
                )

            self.video.delete_data()
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
                    "\n\nIf this error persists please let us know by "
                    "following any of the following options\n"
                    "  - posting on "
                    "https://groups.google.com/g/idtrackerai_users\n"
                    "  - opening an issue at "
                    "https://gitlab.com/polavieja_lab/idtrackerai\n"
                    "  - sending an email to idtrackerai@gmail.com\n"
                    f"Share the log file ({log_file_path}) when "
                    "doing any of the options above"
                )
            return False
        else:
            return True

    def save(self):
        if hasattr(self, "video"):
            self.video.save()
        if hasattr(self, "list_of_blobs"):
            self.list_of_blobs.save(self.video.blobs_path)
        if hasattr(self, "list_of_fragments"):
            self.list_of_fragments.save(self.video.fragments_path)
        if hasattr(self, "list_of_global_fragments"):
            self.list_of_global_fragments.save(
                self.video.global_fragments_path
            )
