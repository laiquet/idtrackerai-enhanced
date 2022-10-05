import logging
from idtrackerai.video import Video
from idtrackerai.animals_detection import AnimalsDetectionAPI
from idtrackerai.crossings_detection import CrossingsDetectionAPI
from idtrackerai.fragmentation import FragmentationAPI
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils.py_utils import CheckSegmentationError

logger = logging.getLogger()


class RunIdTrackerAi:
    def __init__(self, GUI_parameters, *args, **kwargs):
        self.user_parameters = GUI_parameters

    #########################################################
    ## GUI EVENTS ###########################################
    #########################################################

    def print_final_parameters(self):
        logger.info("VIDEO PARAMETERS")

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
        align = max([len(key) for key in keys_to_print])

        for key in keys_to_print:
            logger.info(
                f"[bold]{key:>{align}}[/] = {getattr(self.video_object,key)}",
                extra={"markup": True},
            )

    def track_video(self):
        logger.info("Calling track_video")
        global_success = False
        try:
            # Init tracking manager
            self._step0_init_video_object()
            self.print_final_parameters()
            # exit()
            # self._step1_get_user_defined_parameters()
            # Preprocessing
            # success will be False if there are more blobs than animals and
            # the user asked to check the segmentation consistency
            step2_success = self._step2_pre_processing()
            # Training and identification and post processing
            if step2_success:
                step3_success = self._step3_tracking()
                if step3_success:
                    # This flag is important to register the smoke tests that work
                    global_success = True
                    logger.info("Success")

        except Exception as e:
            self.save()
            if isinstance(e, CheckSegmentationError):
                # Avoid traceback for check_segmentation
                logger.critical(e, exc_info=False)
            else:
                logger.critical(e, exc_info=True)
                logger.info(
                    "\n\nIf this error persists please let us know by\n"
                    "  - posting on https://groups.google.com/g/idtrackerai_users\n"
                    "  - opening an issue at https://gitlab.com/polavieja_lab/idtrackerai\n"
                    "  - sending an email to idtrackerai@gmail.com\n"
                    f"Share the log file ({self.user_parameters['log_file_path']}) when "
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

    def _step0_init_video_object(self):
        logger.info("START: INIT VIDEO OBJECT")
        self.video_object = Video(**self.user_parameters)
        logger.info("FINISH: INIT VIDEO OBJECT")

    def _step2_pre_processing(self):

        logger.info("START: ANIMAL DETECTION")
        self.list_of_blobs = AnimalsDetectionAPI(self.video_object)()
        logger.info("FINISH: ANIMAL DETECTION")

        logger.info("START: CROSSING DETECTION")
        crossings_detector = CrossingsDetectionAPI(
            self.video_object, self.list_of_blobs
        )
        crossings_detector()
        logger.info("FINISH: CROSSING DETECTION")

        logger.info("START: FRAGMENTATION")
        fragmentator = FragmentationAPI(self.video_object, self.list_of_blobs)
        self.list_of_fragments = fragmentator()
        logger.info("FINISH: FRAGMENTATION")
        return True  # This will make the tracking continue

    def _step3_tracking(self):

        tracker = TrackerAPI(
            self.video_object, self.list_of_blobs, self.list_of_fragments
        )

        if self.video_object.track_wo_identification:
            # START: FRAGMENTATION
            logger.info("START: TRACKING WITHOUT IDENTITIES")
            tracker.track_wo_identification()
            logger.info("FINISH: TRACKING WITHOUT IDENTITIES")
            self._final_message = (
                "Tracking without identities finished. "
                "No estimated accuracy computed."
            )
        else:
            if self.video_object.number_of_animals == 1:
                logger.info("START: TRACKING SINGLE ANIMAL")
                tracker.track_single_animal()
                logger.info("FINISH: TRACKING SINGLE ANIMAL")

            else:
                tracker.track_multiple_animals()
                self.list_of_fragments.update_identification_images_dataset()

            logger.info(
                "Estimated accuracy: {}".format(
                    self.video_object.estimated_accuracy
                )
            )

            self.video_object.delete_data()

        return True
