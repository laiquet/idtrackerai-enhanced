import logging
from idtrackerai import Video
from idtrackerai.animals_detection import AnimalsDetectionAPI
from idtrackerai.crossings_detection import CrossingsDetectionAPI
from idtrackerai.fragmentation import FragmentationAPI
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils.py_utils import CheckSegmentationError
import os


class RunIdTrackerAi:
    def __init__(self, GUI_parameters, *args, **kwargs):
        self.user_parameters = GUI_parameters

    def print_final_parameters(self):
        logging.info("VIDEO PARAMETERS")

        keys_to_print = [
            "session",
            "video_paths",
            "intensity_ths",
            "area_ths",
            "number_of_frames",
            "tracking_intervals",
            "number_of_animals",
            "use_bkg",
            "track_wo_identification",
            "use_ROI",
            "check_segmentation",
            "resolution_reduction",
        ]

        params_info = "VIDEO PARAMETERS"

        for key in keys_to_print:
            params_info += (
                f"\n[bold]{key:>{20}}[/] = {getattr(self.video_object,key)}"
            )

        logging.info(params_info, extra={"markup": True})

    def track_video(self):
        logging.info("Calling track_video")
        global_success = False
        try:
            logging.info("START: INIT VIDEO OBJECT")
            self.video_object = Video(**self.user_parameters)
            logging.info("FINISH: INIT VIDEO OBJECT")

            self.print_final_parameters()

            logging.info("START: ANIMAL DETECTION")
            self.list_of_blobs = AnimalsDetectionAPI(self.video_object)()
            logging.info("FINISH: ANIMAL DETECTION")

            logging.info("START: CROSSING DETECTION")
            CrossingsDetectionAPI(self.video_object, self.list_of_blobs)()
            logging.info("FINISH: CROSSING DETECTION")

            logging.info("START: FRAGMENTATION")
            self.list_of_fragments = FragmentationAPI(
                self.video_object, self.list_of_blobs
            )()
            logging.info("FINISH: FRAGMENTATION")

            self.tracking()

            global_success = True
            logging.info("Success")

        except Exception as e:
            self.save()
            if isinstance(e, CheckSegmentationError):
                # Avoid traceback for check_segmentation
                logging.critical(e, exc_info=False)
            else:
                logging.critical(e, exc_info=True)
                lof_file_path = os.path.abspath("idtrackerai-app.log")
                logging.info(
                    "\n\nIf this error persists please let us know by\n"
                    "  - posting on "
                    "https://groups.google.com/g/idtrackerai_users\n"
                    "  - opening an issue at "
                    "https://gitlab.com/polavieja_lab/idtrackerai\n"
                    "  - sending an email to idtrackerai@gmail.com\n"
                    f"Share the log file ({lof_file_path}) when "
                    "doing any of the options above"
                )

        return global_success

    def save(self):
        if hasattr(self, "video_object"):
            self.video_object.save()
        if hasattr(self, "list_of_blobs"):
            self.list_of_blobs.save(self.video_object.blobs_path)
        if hasattr(self, "list_of_fragments"):
            self.list_of_fragments.save(self.video_object.fragments_path)
        if hasattr(self, "list_of_global_fragments"):
            self.list_of_global_fragments.save(
                self.video_object.global_fragments_path,
                self.list_of_fragments.fragments,
            )

    def tracking(self):

        tracker = TrackerAPI(
            self.video_object, self.list_of_blobs, self.list_of_fragments
        )

        if self.video_object.track_wo_identification:
            logging.info("START: TRACKING WITHOUT IDENTITIES")
            tracker.track_wo_identification()
            logging.info("FINISH: TRACKING WITHOUT IDENTITIES")
            logging.info(
                "Tracking without identities finished, "
                "no estimated accuracy computed."
            )
        else:
            if self.video_object.number_of_animals == 1:
                logging.info("START: TRACKING SINGLE ANIMAL")
                tracker.track_single_animal()
                logging.info("FINISH: TRACKING SINGLE ANIMAL")

            else:
                tracker.track_multiple_animals()
                self.list_of_fragments.update_identification_images_dataset()

            logging.info(
                f"Estimated accuracy: {self.video_object.estimated_accuracy}"
            )

            self.video_object.delete_data()
