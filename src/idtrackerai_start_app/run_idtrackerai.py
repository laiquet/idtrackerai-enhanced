import logging
import sys
from os import cpu_count
from pathlib import Path
from shutil import copy

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Video
from idtrackerai.animals_detection import animals_detection_API
from idtrackerai.crossings_detection import crossings_detection_API
from idtrackerai.fragmentation import fragmentation_API
from idtrackerai.postprocess import trajectories_API
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils import CustomError, conf


class RunIdTrackerAi:
    def __init__(self, user_parameters: dict):
        # Set the number of jobs accordingly to the computer number of CPUs
        n_jobs = user_parameters["number_of_parallel_workers"]
        computer_CPUs = cpu_count()
        if computer_CPUs is None:
            n_jobs = max(0, n_jobs)
        else:
            if n_jobs == 0:
                n_jobs = (computer_CPUs + 1) // 2
            elif n_jobs < 0:
                n_jobs = computer_CPUs + n_jobs
        user_parameters["number_of_parallel_workers"] = n_jobs
        logging.info(f"Number of parallel jobs: {n_jobs}")

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
            sys.exit()

        self.user_parameters = {
            param: getattr(conf, param) for param in mandatory_parameters
        }

        # add optional args
        self.user_parameters["bkg_model"] = getattr(conf, "bkg_model", None)

        self.video: Video
        self.list_of_blobs: ListOfBlobs
        self.list_of_fragments: ListOfFragments
        self.list_of_global_fragments: ListOfGlobalFragments

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
            success = True

        except CustomError as error:
            logging.error(
                "An error occurred, saving data before "
                "printing traceback and exiting the program"
            )
            self.save()
            logging.critical(error, exc_info=False)
            success = False
        except Exception as error:
            logging.error(
                "An error occurred, saving data before "
                "printing traceback and exiting the program"
            )
            self.save()

            logging.critical(error, exc_info=True)
            log_file_path = Path("idtrackerai.log").resolve()
            logging.info(
                "\n\nIf this error persists please let us know by "
                "following any of the following options\n"
                "  - posting on "
                "https://groups.google.com/g/idtrackerai_users\n"
                "  - opening an issue at "
                "https://gitlab.com/polavieja_lab/idtrackerai\n"
                "  - sending an email to idtrackerai@gmail.com\n"
                "Share the log file (%s) when "
                "doing any of the options above",
                log_file_path,
            )
            success = False

        copy(Path("idtrackerai.log"), self.video.session_folder / "idtrackerai.log")
        return success

    def save(self):
        if hasattr(self, "video"):
            self.video.save()
        if hasattr(self, "list_of_blobs"):
            self.list_of_blobs.save(self.video.blobs_path)
        if hasattr(self, "list_of_fragments"):
            self.list_of_fragments.save(self.video.fragments_path)
        if hasattr(self, "list_of_global_fragments"):
            self.list_of_global_fragments.save(self.video.global_fragments_path)
