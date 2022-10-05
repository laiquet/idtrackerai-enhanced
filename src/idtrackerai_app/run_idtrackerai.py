import numpy as np
import os
import logging
import traceback
from confapp import conf


from idtrackerai.animals_detection.segmentation_utils import compute_background

from idtrackerai.video import Video
from idtrackerai.animals_detection import AnimalsDetectionAPI
from idtrackerai.crossings_detection import CrossingsDetectionAPI
from idtrackerai.fragmentation import FragmentationAPI
from idtrackerai.tracker.tracker import TrackerAPI


logger = logging.getLogger(__name__)


class RunIdTrackerAi:

    ERROR_MESSAGE_DEFAULT = (
        "\n \nIf this error persists please open an issue at "
        "https://gitlab.com/polavieja_lab/idtrackerai or "
        "send an email to idtrackerai@gmail.com. "
        "Check the log file idtrackerai-app.log in your "
        "working directory and attach it to the issue."
    )

    SEGMENTATION_CHECK_FINAL_MESSAGE = (
        "Readjust the segmentation parameters and track the video again."
    )

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
            logger.critical(e, exc_info=True)
            # print(traceback.format_exc())
            self.save()

        return global_success

    def save(self):
        logger.info("Saving objects from base_idtrackerai")
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

    def __output_segmentation_consistency_warning(self, outfile_path):
        self.warning(
            "On some frames it was found more blobs than "
            "animals, "
            "you can find the index of these frames in the file:"
            f"<p>{outfile_path}</p>"
            "<p>Please readjust the segmentation parameters and "
            "press 'Track video' again.</p>",
            "Found more blobs than animals",
        )
        self._final_message = self.SEGMENTATION_CHECK_FINAL_MESSAGE

    def _step2_pre_processing(self):

        logger.info("START: ANIMAL DETECTION")
        animals_detector = AnimalsDetectionAPI(self.video_object)
        self.list_of_blobs = animals_detector()
        # Check segmentation consistency
        segmentation_consistent = animals_detector.check_segmentation()
        if (
            not segmentation_consistent
            and self.user_parameters["check_segmentation"]
        ):
            outfile_path = animals_detector.save_inconsistent_frames()
            self.save()  # saves video_object
            self.__output_segmentation_consistency_warning(outfile_path)
            return False  # This will make the tracking finish
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

            self._final_message = (
                "Tracking finished with {0:.2f} "
                "estimated accuracy.".format(
                    self.video_object.estimated_accuracy * 100
                )
            )
        return True
