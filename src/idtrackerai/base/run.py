import logging
from shutil import copy

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Session
from idtrackerai.utils import LOG_FILE_PATH, Timer

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
            general_timer = Timer("Tracking session")
            general_timer.start()

            self.session.prepare_tracking()
            self.session.timers["Tracking session"] = general_timer

            self.save()
            with self.session.new_timer("Animal detection"):
                self.list_of_blobs = animals_detection_API(self.session)

            self.save()

            with self.session.new_timer("Crossing detection"):
                crossings_detection_API(self.session, self.list_of_blobs)

            self.save()

            with self.session.new_timer("Fragmentation"):
                self.list_of_fragments, self.list_of_global_fragments = (
                    fragmentation_API(self.session, self.list_of_blobs)
                )

            self.save()

            with self.session.new_timer("Tracking"):
                self.list_of_fragments = tracker_API(
                    self.session,
                    self.list_of_blobs,
                    self.list_of_fragments,
                    self.list_of_global_fragments,
                )

            self.save()

            with self.session.new_timer("Trajectories creation"):
                trajectories_API(
                    self.session,
                    self.list_of_blobs,
                    self.list_of_global_fragments.single_global_fragment,
                    self.list_of_fragments,
                )

            self.session.timers["Tracking session"].finish()
            self.session.save()

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

        except (Exception, KeyboardInterrupt) as error:

            if (
                hasattr(self, "session")
                and hasattr(self.session, "session_folder")
                and self.session.session_folder.is_dir()
                and LOG_FILE_PATH.is_file()
            ):
                # we add the path where we would like to have a copy of the log
                # TODO when Python >= 3.11 use Exception.add_note()
                error.log_path = self.session.session_folder / LOG_FILE_PATH.name  # type: ignore
            raise error

        if (
            hasattr(self, "session")
            and hasattr(self.session, "session_folder")
            and self.session.session_folder.is_dir()
            and LOG_FILE_PATH.is_file()
        ):
            log_copy_path = self.session.session_folder / LOG_FILE_PATH.name
            copy(LOG_FILE_PATH, log_copy_path)
            logging.info(f"Log file copied to {log_copy_path}")
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
