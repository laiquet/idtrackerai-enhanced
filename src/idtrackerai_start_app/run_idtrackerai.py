import logging
from shutil import copy

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Video
from idtrackerai.animals_detection import animals_detection_API
from idtrackerai.crossings_detection import crossings_detection_API
from idtrackerai.fragmentation import fragmentation_API
from idtrackerai.postprocess import trajectories_API
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils import LOG_FILE_PATH, CustomError


class RunIdTrackerAi:
    video: Video
    list_of_blobs: ListOfBlobs
    list_of_fragments: ListOfFragments
    list_of_global_fragments: ListOfGlobalFragments

    def __init__(self, video: Video):
        self.video = video

    def track_video(self) -> bool:
        try:
            self.video.prepare_tracking()

            self.save()

            self.list_of_blobs = animals_detection_API(self.video)

            self.save()

            crossings_detection_API(self.video, self.list_of_blobs)

            self.save()

            self.list_of_fragments, self.list_of_global_fragments = fragmentation_API(
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
                    if self.list_of_global_fragments.no_global_fragment:
                        raise CustomError(
                            "There are no Global Fragments long enough to be candidates"
                            " for accumulation, thus it is not possible to train the"
                            " identification networks. The video has to contain longer"
                            " slices where all animals are visible without crossings."
                        )
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
                    "Tracked without identities, no estimated accuracy available."
                )
            else:
                logging.info(f"Estimated accuracy: {self.video.estimated_accuracy:.4%}")

            self.video.delete_data()
            self.video.compress_data()
            logging.info("[green]Success", extra={"markup": True})
            success = True

        except Exception as error:
            logging.error(
                "An error occurred, saving data before "
                "printing traceback and exiting the program"
            )
            self.save()
            raise error

        if hasattr(self, "video") and LOG_FILE_PATH.is_file():
            copy(LOG_FILE_PATH, self.video.session_folder / LOG_FILE_PATH.name)
        return success

    def save(self):
        try:
            if hasattr(self, "video"):
                self.video.save()
            if hasattr(self, "list_of_blobs"):
                self.list_of_blobs.save(self.video.blobs_path)
            if hasattr(self, "list_of_fragments"):
                self.list_of_fragments.save(self.video.fragments_path)
            if hasattr(self, "list_of_global_fragments"):
                self.list_of_global_fragments.save(self.video.global_fragments_path)
        except Exception as exc:
            logging.error("Error while saving data: %s", exc)
