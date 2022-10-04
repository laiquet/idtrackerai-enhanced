import numpy as np, os, logging

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

        [
            "number_of_animals",
            "intensity_ths",
            "area_ths",
            "check_segmentation",
            "tracking_interval",
            "ROI_list",
            "use_bkg",
            "bkg_model",
            "resolution_reduction",
            "track_wo_identification",
            "setup_points",
            "knowledge_transfer_folder",
            "identity_transfer",
            "identification_image_size",
        ]

    #########################################################
    ## GUI EVENTS ###########################################
    #########################################################

    def track_video(self):
        logger.info("Calling track_video")
        try:
            # Init tracking manager
            self._step0_init_video_object()
            self._step1_get_user_defined_parameters()
            exit()
            # Preprocessing
            # success will be False if there are more blobs than animals and
            # the user asked to check the segmentation consistency
            success = self._step2_pre_processing()
            # Training and identification and post processing
            if success:
                success = self._step3_tracking()
            if success:
                # This flag is important to register the smoke tests that work
                logger.info("Success")

        except Exception as e:
            self.save()
            # print(e)
            logger.critical(e, exc_info=True)

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
        # if not os.path.exists(self.video_path):
        #     raise Exception(
        #         "The video you are trying to track does not exist or the "
        #         f"path to the video is wrong: {self.video_path}"
        #     )

        # INIT AND POPULATE VIDEO OBJECT WITH PARAMETERS
        # self.__get_tracking_interval()
        # logger.info(
        #     f"Tracking interval precomputed as {self._tracking_interval}"
        # )

        logger.info("START: INIT VIDEO OBJECT")
        self.video_object = Video(
            video_path=self.GUI_parameters.get("video_paths"),
            open_multiple_files=self.GUI_parameters.get("open-multiple-files"),
            tracking_intervals=self.GUI_parameters.get("tracking_intervals"),
        )
        logger.info("FINISH: INIT VIDEO OBJECT")
        self.video_object.create_session_folder(
            self.GUI_parameters.get("session")
        )

    # def __get_tracking_interval(self):
    #     if self._multiple_range.value and self._rangelst.value:
    #         try:
    #             self._tracking_interval = eval(self._rangelst.value)
    #         except Exception as e:
    #             logger.fatal(e, exc_info=True)
    #             self._tracking_interval = [self._range.value]
    #     else:
    #         self._tracking_interval = [self._range.value]

    def __get_bkg_model(self):
        if self._bgsub.value:
            if self._background_img is None:
                # Asked for background subtraction but it is not computed
                logger.info("Computing the background model")
                self._mask_img = self.create_mask(
                    self.video_object.original_height,
                    self.video_object.original_width,
                )
                self._background_img = compute_background(
                    self.video_object.video_paths,
                    self._mask_img,
                    self.video_object.episodes,
                )
            else:
                logger.info("Storing previously computed background model")
        else:
            # Did not ask for background subtraction
            logger.info("No background model computed")
            self._background_img = None

    def __get_mask(self):
        if self._applyroi:
            self._mask_img = self.create_mask(
                self.video_object.original_height,
                self.video_object.original_width,
            )
        else:
            self._mask_img = np.ones(
                (
                    self.video_object.original_height,
                    self.video_object.original_width,
                )
            )

    def _step1_get_user_defined_parameters(self):

        self.__get_mask()
        # The computation of the background has a computation of the mask
        # TODO: Separate better mask and bkg
        self.__get_bkg_model()
        # TODO: Separate user defined parameters and advanced parameters
        # There are other parameters that come form the local_settings.py
        # It would be great to store them all in a singe json file so we
        # can check all the parameters used for tracking
        user_defined_parameters = {
            "number_of_animals": int(self._number_of_animals.value),
            "min_threshold": self._intensity.value[0],
            "max_threshold": self._intensity.value[1],
            "min_area": self._area.value[0],
            "max_area": self._area.value[1],
            "check_segmentation": self._chcksegm.value,
            "tracking_interval": self._tracking_interval,
            "apply_ROI": self._applyroi.value,
            "rois": self._roi.value,
            "mask": self._mask_img,
            "subtract_bkg": self._bgsub.value,
            "bkg_model": self._background_img,
            "resolution_reduction": self._resreduct.value,
            "track_wo_identification": self._no_ids.value,
            "setup_points": self.create_setup_poitns_dict(),
            "sigma_gaussian_blurring": conf.SIGMA_GAUSSIAN_BLURRING,
            "knowledge_transfer_folder": conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN,
            "identity_transfer": False,
            "identification_image_size": None,
        }

        if conf.IDENTITY_TRANSFER:
            # TODO: the identification_image_size is not really passed by
            # the used but inferred from the knowledge transfer folder
            (
                user_defined_parameters["identity_transfer"],
                user_defined_parameters["identification_image_size"],
            ) = TrackerAPI.check_if_identity_transfer_is_possible(
                user_defined_parameters["number_of_animals"],
                conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN,
            )

        self.video_object._user_defined_parameters = user_defined_parameters

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
        if not segmentation_consistent and self._chcksegm.value:
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

        if self.video_object.user_defined_parameters[
            "track_wo_identification"
        ]:
            # START: FRAGMENTATION
            logger.info("START: TRACKING WITHOUT IDENTITIES")
            tracker.track_wo_identification()
            logger.info("FINISH: TRACKING WITHOUT IDENTITIES")
            self._final_message = (
                "Tracking without identities finished. "
                "No estimated accuracy computed."
            )
        else:
            if (
                self.video_object.user_defined_parameters["number_of_animals"]
                == 1
            ):
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
