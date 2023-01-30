import logging
from pathlib import Path
from shutil import copy

from idtrackerai import Video
from idtrackerai.animals_detection import animals_detection_API
from idtrackerai.crossings_detection import crossings_detection_API
from idtrackerai.fragmentation import fragmentation_API
from idtrackerai.postprocess import trajectories_API
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils import CustomError, conf, pprint_dict


class RunIdTrackerAi:
    def __init__(self, user_parameters: dict):
        conf.set_dict(user_parameters)

        mandatory_parameters = (
            "video_paths",
            "number_of_animals",
            "intensity_ths",
            "area_ths",
            "output_dir",
            "session",
            "tracking_intervals",
            "resolution_reduction",
            "roi_list",
            "use_bkg",
            "track_wo_identities",
            "sigma_gaussian_blurring",
            "check_segmentation",
            "identity_transfer",
            "knowledge_transfer_folder",
        )

        missing_parameters = [
            param for param in mandatory_parameters if not hasattr(conf, param)
        ]

        if missing_parameters:
            logging.error(f"The following parameters are missing: {missing_parameters}")
            exit()

        self.user_parameters = {
            param: getattr(conf, param) for param in mandatory_parameters
        }

        # add optional args
        self.user_parameters["ROI_mask"] = getattr(conf, "ROI_mask", None)
        self.user_parameters["bkg_model"] = getattr(conf, "bkg_model", None)

    def track_video(self) -> bool:
        try:
            self.video = Video(**self.user_parameters)

            self.save()

            self.list_of_blobs = animals_detection_API(self.video)

            self.save()

            crossings_detection_API(self.video, self.list_of_blobs)

            self.save()

            (self.list_of_fragments, self.list_of_global_fragments) = fragmentation_API(
                self.video, self.list_of_blobs
            )
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
                    if self.list_of_global_fragments.single_global_fragment:
                        tracker.track_single_global_fragment_video()
                    else:
                        self.list_of_fragments = tracker.track_with_identities()
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
                logging.info(f"Estimated accuracy: {self.video.estimated_accuracy:.4%}")

            self.video.delete_data()
            logging.info("Success")
            copy(Path("idtrackerai.log"), self.video.session_folder / "idtrackerai.log")

        except Exception as e:
            logging.error(
                "An error occurred, saving data before "
                "printing traceback and exiting the program"
            )
            self.save()
            if isinstance(e, CustomError):
                # Avoid traceback for custom errors
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
            self.list_of_global_fragments.save(self.video.global_fragments_path)
