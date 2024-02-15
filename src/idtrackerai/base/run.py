import logging
from shutil import copy

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Session
from idtrackerai.utils import LOG_FILE_PATH

from .animals_detection import animals_detection_API
from .crossings_detection import crossings_detection_API
from .fragmentation import fragmentation_API
from .postprocess import trajectories_API
from .tracker import tracker_API


class RunIdTrackerAi:
    session: Session
    list_of_blobs: ListOfBlobs
    list_of_fragments: ListOfFragments
    list_of_global_fragments: ListOfGlobalFragments

    def __init__(self, session: Session):
        self.session = session

    def track_video(self) -> bool:
        try:
            self.session.prepare_tracking()

            self.save()

            self.list_of_blobs = animals_detection_API(self.session)

            self.save()

            crossings_detection_API(self.session, self.list_of_blobs)

            self.save()

            self.list_of_fragments, self.list_of_global_fragments = fragmentation_API(
                self.session, self.list_of_blobs
            )

            self.save()

            self.list_of_fragments = tracker_API(
                self.session,
                self.list_of_blobs,
                self.list_of_fragments,
                self.list_of_global_fragments,
            )

            self.save()

            trajectories_API(
                self.session,
                self.list_of_blobs,
                self.list_of_global_fragments.single_global_fragment,
                self.list_of_fragments,
            )

            if self.session.track_wo_identities:
                logging.info(
                    "Tracked without identities, no estimated accuracy available."
                )
            else:
                logging.info(
                    f"Estimated accuracy: {self.session.estimated_accuracy:.4%}"
                )

            self.session.delete_data()
            self.session.compress_data()
            logging.info("[green]Success", extra={"markup": True})
            success = True

        except Exception as error:
            logging.error(
                "An error occurred, saving data before "
                "printing traceback and exiting the program"
            )
            self.save()

            if (
                hasattr(self, "session")
                and hasattr(self.session, "session_folder")
                and LOG_FILE_PATH.is_file()
            ):
                copy(LOG_FILE_PATH, self.session.session_folder / LOG_FILE_PATH.name)

            raise error

        if (
            hasattr(self, "session")
            and hasattr(self.session, "session_folder")
            and LOG_FILE_PATH.is_file()
        ):
            copy(LOG_FILE_PATH, self.session.session_folder / LOG_FILE_PATH.name)
        return success

    def save(self):
        try:
            if hasattr(self, "session") and hasattr(self.session, "session_folder"):
                self.session.save()
            if hasattr(self, "list_of_blobs"):
                self.list_of_blobs.save(self.session.blobs_path)
            if hasattr(self, "list_of_fragments"):
                self.list_of_fragments.save(self.session.fragments_path)
            if hasattr(self, "list_of_global_fragments"):
                self.list_of_global_fragments.save(self.session.global_fragments_path)
        except Exception as exc:
            logging.error("Error while saving data: %s", exc)
