import json
import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from importlib import metadata
from itertools import count, pairwise
from math import sqrt
from os import cpu_count
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import cv2
import h5py
import numpy as np

from . import IdtrackeraiError
from .utils import (
    Episode,
    LengthCalibration,
    Timer,
    assert_all_files_exist,
    assert_knowledge_transfer_is_possible,
    build_ROI_mask_from_list,
    create_dir,
    json_default,
    json_object_hook,
    remove_dir,
    resolve_path,
    track,
)


class Session:
    """
    A class containing the main features of the session.

    This class includes properties of the video by itself, user defined
    parameters for the tracking, and other properties that are generated
    throughout the tracking process.

    We use this class as a storage of data coming from different processes.
    However, this is bad practice and it will change in the future.
    """

    velocity_threshold: float
    erosion_kernel_size: int
    ratio_accumulated_images: float
    individual_fragments_stats: dict
    session_folder: Path
    setup_points: dict[str, list[tuple[int, int]]]
    median_body_length: float
    """median of the diagonals of individual blob's bounding boxes"""
    first_frame_first_global_fragment: int = -1
    identities_groups: dict
    """Named groups of identities stored in the validation GUI.
    If `exclusive ROI`, the identities of each region will be saved here"""
    length_calibrations: list[LengthCalibration]
    """List of length calibrations containing two points (in pixels units) and
    the real distance between these two defined by the user in the Validator."""
    episodes: list[Episode]
    """Indicates the starting and ending frames of each video episode.
    Video episodes are used for parallelization of some processes"""
    width: int
    """Video width in pixels"""
    height: int
    """Video height in pixels"""
    frames_per_second: float
    """Video frame rate in frames per second obtained by OpenCV from the
    video file"""
    accumulation_statistics_data: dict[str, list]
    version: str
    """Version of idtracker.ai"""
    number_of_error_frames: int = -1
    """The number of frames with more blobs than animals. Set on animals_detection."""
    estimated_accuracy: float | None = None
    identities_labels: list[str] | None = None
    """A list with a name for every identity. Defined and used in validator"""
    background_from_segmentation_gui: Path | None = None
    """Path to the background computed by the segmentation app. It is reused at tracking time"""

    video_paths: list[Path] = []
    """List of paths to the different files the video is composed of.
    If the video is a single file, the list will have length 1"""
    number_of_animals: int = 0
    intensity_ths: None | Sequence[float] = None
    area_ths: None | Sequence[float] = None
    # bkg_model: None | np.ndarray = None
    name: str = ""
    output_dir: Path | None | str = None
    tracking_intervals: list | None = None
    resolution_reduction: float | None = None
    roi_list: list[str] | str | None = None
    use_bkg: bool = False
    knowledge_transfer_folder: None | Path = None
    check_segmentation: bool = False
    identity_transfer: bool = False
    track_wo_identities: bool = False
    frames_per_episode: int = 500
    background_subtraction_stat: Literal["median", "mean", "max", "min"] = "median"
    number_of_frames_for_background: int = 50
    number_of_parallel_workers: int = 0
    data_policy: Literal[
        "trajectories", "validation", "knowledge_transfer", "idmatcher.ai", "all"
    ] = "idmatcher.ai"
    id_image_size: list[int] = []
    """ Shape of the Blob's identification images (width, height, n_channels)"""
    trajectories_formats: Sequence[Literal["h5", "npy", "csv", "pickle"]] = [
        "h5",
        "npy",
        "csv",
    ]
    """A sequence of strings defining in which formats the trajectories should be saved"""
    exclusive_rois: bool = False
    """Treat each separate ROI as closed identities groups"""
    identity_transfer_succeeded: bool = False
    "True if the identity transfer has been done successfully"
    bounding_box_images_in_ram: bool = False
    "Keep bounding box images on RAM and until used, never write them on disk"
    last_validated: datetime | None = None
    "Last time this session was validated using the Validator"
    silhouette_score: float | None = None
    "Silhouette score reached at the end of the contrastive step"

    def set_parameters(self, reset: bool = False, **parameters) -> set[str]:
        """Sets parameters to self only if they are present in the class annotations.

        Parameters
        ----------
        reset : bool, optional
            If True, resets the session parameters before setting new ones, by default False.
        **parameters : dict
            Arbitrary keyword arguments representing the parameters to set.

        Returns
        -------
        set[str]
            The set of non recognized parameters names.
        """
        if reset:
            logging.info("Restarting Session parameters")
            for key in list(self.__dict__.keys()):
                if key in self.__class__.__annotations__:
                    del self.__dict__[key]
        non_recognized_parameters: set[str] = set()
        for param, value in parameters.items():
            lower_param = param.lower()
            if lower_param in self.__class__.__annotations__:
                setattr(self, lower_param, value)
            else:
                non_recognized_parameters.add(param)
        return non_recognized_parameters

    def prepare_tracking(self) -> None:
        """Initializes the session object, checking all parameters"""
        logging.debug("Initializing Session")
        try:
            self.version = metadata.version("idtrackerai")
        except metadata.PackageNotFoundError:
            raise IdtrackeraiError("idtrackerai package is not installed")

        if not isinstance(self.video_paths, list):
            video_paths = [self.video_paths]
        else:
            video_paths = self.video_paths
        self.assert_video_paths(video_paths)
        self.video_paths = [resolve_path(path) for path in video_paths]
        logging.info(
            "Setting video paths to:\n    " + "\n    ".join(map(str, self.video_paths))
        )

        if self.area_ths is None:
            raise IdtrackeraiError("Missing area thresholds parameter")

        if self.intensity_ths is None:
            raise IdtrackeraiError("Missing intensity thresholds parameter")

        if self.knowledge_transfer_folder is not None:
            self.knowledge_transfer_folder = resolve_path(
                self.knowledge_transfer_folder
            )
            if not self.knowledge_transfer_folder.exists():
                raise IdtrackeraiError(
                    f'Knowledge transfer folder "{self.knowledge_transfer_folder}" not'
                    " found"
                )

            if (
                self.knowledge_transfer_folder.is_dir()
                and self.knowledge_transfer_folder.name.startswith("session_")
            ):
                if (self.knowledge_transfer_folder / "accumulation_0").is_dir():
                    self.knowledge_transfer_folder /= "accumulation_0"
                else:
                    self.knowledge_transfer_folder /= "accumulation"

            self.id_image_size, self.resolution_reduction = (
                assert_knowledge_transfer_is_possible(
                    self.knowledge_transfer_folder, self.n_animals
                )
            )

        self.width, self.height, self.frames_per_second = (
            self.get_info_from_video_paths(self.video_paths)
        )
        self.number_of_frames, _, self.tracking_intervals, self.episodes = (
            self.get_processing_episodes(
                self.video_paths, self.frames_per_episode, self.tracking_intervals
            )
        )

        trackable_n_frames = sum(
            interv[1] - interv[0] for interv in self.tracking_intervals
        )
        if trackable_n_frames != self.number_of_frames:
            logging.info(
                f"The session has {self.number_of_frames} frames from which "
                f"{trackable_n_frames} will be tracked "
                f"({self.number_of_episodes} episodes)"
            )
        else:
            logging.info(
                f"The session has {self.number_of_frames} "
                f"frames ({self.number_of_episodes} episodes)"
            )
        if len(self.episodes) < 10:
            for episode in self.episodes:
                video_name = episode.video_path.name
                logging.info(
                    f"\tEpisode {episode.index}, frames ({episode.local_start} "
                    f"-> {episode.local_end}) of /{video_name}"
                )
        assert self.number_of_episodes > 0

        if self.output_dir is None:
            self.output_dir = self.video_paths[0].parent
        self.output_dir = resolve_path(self.output_dir)

        if not self.name:
            self.name = "&".join(path.stem for path in self.video_paths)
            logging.info('No session name provided, assigning name "%s"', self.name)

            if (self.output_dir / f"session_{self.name}").exists():
                # add a counter in sessions with default name
                for index in count(1):
                    name = self.name + f"_{index}"
                    if not (self.output_dir / f"session_{name}").exists():
                        break
                else:
                    raise RuntimeError
                logging.info(
                    'A session with the assigned name ("%s") already exists, renaming'
                    ' current session to "%s"',
                    self.name,
                    name,
                )
                self.name = name

        self.session_folder = self.output_dir / f"session_{self.name}"

        create_dir(self.session_folder)
        create_dir(self.preprocessing_folder)

        self.ROI_mask = build_ROI_mask_from_list(self.roi_list, self.width, self.height)

        if isinstance(self.id_image_size, int):
            self.id_image_size = [self.id_image_size, self.id_image_size, 1]
        elif not self.id_image_size:  # if it is None or empty tuple or list...
            self.id_image_size = []

        if self.number_of_parallel_workers <= 0:
            computer_CPUs = cpu_count()
            if computer_CPUs is not None:
                if self.number_of_parallel_workers == 0:
                    self.number_of_parallel_workers = min((computer_CPUs + 1) // 2, 4)
                elif self.number_of_parallel_workers < 0:
                    self.number_of_parallel_workers += computer_CPUs
            else:
                logging.warning(
                    "Could not determine the number of CPUs in the computer."
                    " Setting the number of parallel jobs to 1"
                )
                self.number_of_parallel_workers = 1
        logging.info("Number of parallel jobs: %d", self.number_of_parallel_workers)

        if self.number_of_animals == 0 and not self.track_wo_identities:
            raise IdtrackeraiError(
                "Cannot track with an undefined number of animals (n_animals = 0)"
                " when tracking with identities"
            )

        if (
            self.background_from_segmentation_gui is not None
            and self.background_from_segmentation_gui.is_file()
        ):
            # If the background was computed by the segmentation GUI, we move it to the final location
            self.background_path.unlink(missing_ok=True)
            self.background_from_segmentation_gui.rename(self.background_path)

        self.identities_groups = {}
        self.setup_points = {}

        # Processes timers
        self.timers: dict[str, Timer] = {}

    def save(self) -> None:
        """Saves the instantiated Session object"""
        logging.info(f"Saving Session object in {self.path_to_session}", stacklevel=2)
        dict_to_save = (self.defaults() | vars(self)).copy()
        dict_to_save.pop("episodes", None)
        dict_to_save.pop("output_dir", None)
        dict_to_save.pop("background_from_segmentation_gui", None)
        self.path_to_session.write_text(
            json.dumps(dict_to_save, default=json_default, indent=4)
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
        video_paths_dir: Path | str | None = None,
        allow_not_found_video_files: bool = True,
    ) -> "Session":
        """Load a session object stored in a JSON file

        Parameters
        ----------
        path : Path | str
            Path to the session JSON file to load from or to the session folder containing the JSON file.
        video_paths_dir : Path | None, optional
            Folder containing the video paths. If None (the default), they are expected to be in:

            - In the same location where they were during the tracking
            - In the parent folder of the JSON file being loaded
            - In the double-parent folder of the JSON file being loaded
            - In the parent folder of the original path of the JSON file
            - In the double-parent folder of the original path of the JSON file
            - In the current working directory

        allow_not_found_video_files : bool, optional
            If False, an IdtrackeraiError exception is raised if the video files couldn't be found, by default True.

        Returns
        -------
        Session
            Instance of :class:`Session` from the loaded JSON file.

        Raises
        ------
        FileNotFoundError
            If the JSON couldn't be found.
        IdtrackeraiError
            If the video files couldn't be found and `allow_not_found_video_files` is `True`.
        ValueError
            If the input file is not JSON readable.
            If the session being loaded is from version 4 or lower of idtracker.ai.
        """
        path = resolve_path(path)
        logging.info(f"Loading Session from {path}", stacklevel=2)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        if not path.is_file():
            possible_files = ("session.json", "video_object.json", "video_object.npy")
            for file in possible_files:
                if (path / file).is_file():
                    path /= file
                    break
            else:
                raise FileNotFoundError(
                    f"Session parameterfiles not found in {path}"
                    if path.name.startswith("session_")
                    else f"The path {path} is not a session folder, session folders start with 'session_'"
                )

        if path.suffix == ".npy":
            raise ValueError(
                f"The file {path} is no longer supported by idtrackerai "
                "v6 (or higher) since it was generated by v4 (or lower)"
            )

        # Avoid loading huge video files by mistake
        if path.stat().st_size > 5000000:  # 5MB
            raise ValueError(
                f"{path} takes {path.stat().st_size / (1024**2):.1f} MB, it does not seem"
                " like a session json file"
            )

        try:
            with open(path, encoding="utf_8") as file:
                session_dict: dict[str, Any] = json.load(
                    file, object_hook=json_object_hook
                )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f'The file "{path}" is not JSON readable. Original JSON error message: {exc}'
            ) from exc
        if "original_width" in session_dict:
            session_dict["width"] = session_dict["original_width"]

        if "original_height" in session_dict:
            session_dict["height"] = session_dict["original_height"]

        if "n_animals" not in session_dict and "number_of_animals" in session_dict:
            session_dict["n_animals"] = session_dict["number_of_animals"]

        session_dict["video_paths"] = [
            resolve_path(pth) for pth in session_dict["video_paths"]
        ]

        if "session" in session_dict and "name" not in session_dict:
            # backward compatibility
            session_dict["name"] = session_dict.pop("session")

        session_dict["timers"] = {
            name: Timer.from_dict(timer_dict)
            for name, timer_dict in session_dict.get("timers", {}).items()
        }

        # format timers and Paths
        for key, value in session_dict.items():
            if key.endswith("_timer") and isinstance(value, dict):
                # <=5.2.11 compatibility
                session_dict[key] = Timer.from_dict(value)
            if key.endswith("_folder") and isinstance(value, str):
                session_dict[key] = resolve_path(value)

        if "general_timer" in session_dict:
            # This is the only timer we currently use after tracking,
            # so we want to recover it if it was saved in a previous version style
            session_dict["timers"]["Tracking session"] = session_dict["general_timer"]

        if session_dict.get("length_calibrations"):
            session_dict["length_calibrations"] = [
                LengthCalibration.from_dict(value)
                for value in session_dict["length_calibrations"]
            ]

        session_dict["last_validated"] = (
            datetime.fromisoformat(session_dict["last_validated"])
            if session_dict.get("last_validated") is not None
            else None
        )

        session = cls.__new__(cls)
        session.__dict__.update(session_dict)

        try:
            session.update_paths(path.parent, video_paths_dir)
        except FileNotFoundError as exc:
            if not allow_not_found_video_files:
                # We raise an IdtrackeraiError to distinguish from the
                # FileNotFoundError raised when the .json file is not found
                raise IdtrackeraiError() from exc

        try:
            _, _, _, session.episodes = session.get_processing_episodes(
                session.video_paths,
                session.frames_per_episode,
                session.tracking_intervals,
            )
        except AttributeError:
            logging.warning(
                "Could not load video episodes probably due to loading an old version"
                " session"
            )
        except (IdtrackeraiError, FileNotFoundError) as exc:
            logging.warning("Could not load video episodes. %s", str(exc))

        return session

    def new_timer(self, name: str) -> Timer:
        """Generates, saves and returns a Timer"""
        timer = Timer(name)
        self.timers[name] = timer
        return timer

    def __str__(self) -> str:
        return f"<session {self.session_folder}>"

    def set_id_image_size(self, median_body_length: float) -> None:
        """Sets the :attr:`median_body_length` and computes the appropiate identification image size and resolution reduction when needed and taking into account the existing user defined values.

        Parameters
        ----------
        median_body_length : float
            Median body length of all individual blobs in the video
        """
        max_size = 80  # has to be even
        self.median_body_length = median_body_length
        auto_size = int(median_body_length / sqrt(2) + 0.5)
        auto_size += auto_size % 2  # nearest even integer
        logging.info(f"The automatic identification image size is {auto_size}")

        if not self.id_image_size:
            if self.resolution_reduction is None:
                if auto_size > max_size:
                    self.resolution_reduction = np.round(max_size / auto_size, 2)
                    self.id_image_size = [max_size, max_size, 1]
                    logging.info(
                        f"Since this is bigger than {max_size}, the resolution "
                        f"reduction is set to {self.resolution_reduction} to diminish"
                        f" this image size to {max_size} pixels"
                    )
                else:
                    logging.info("No resolution reduction required")
                    self.resolution_reduction = 1
                    self.id_image_size = [auto_size, auto_size, 1]
            else:
                scaled_size = int(auto_size * self.resolution_reduction + 0.5)
                scaled_size += scaled_size % 2
                logging.info(
                    f"The resolution reduction is fixed to {self.resolution_reduction}"
                    " so the automatic identification image size is rescaled "
                    f"to {scaled_size}"
                )
                self.id_image_size = [scaled_size, scaled_size, 1]
        else:
            logging.info(
                f"But the identification image size is fixed by the user to {self.id_image_size}"
            )
            if self.resolution_reduction is None:
                if auto_size > self.id_image_size[0]:
                    self.resolution_reduction = np.round(
                        self.id_image_size[0] / auto_size, 2
                    )
                    logging.info(
                        f"Since animals are bigger than the image size defined by the"
                        " user, the resolution reduction is set to "
                        f"{self.resolution_reduction} to fit them"
                    )
                else:
                    logging.info("No resolution reduction required")
                    self.resolution_reduction = 1
            else:
                logging.info(
                    "The identification image size and the resolution "
                    "reduction are both fixed by the user."
                )

        logging.info(
            f"Identification image size set to {self.id_image_size},"
            f" resolution reduction factor set to {self.resolution_reduction}"
        )
        if self.id_image_size[0] > max_size:
            logging.warning(
                f"The identification image size is bigger than {max_size}. "
                "This can unnecessarily slow down the models training"
            )

    @property
    def n_animals(self) -> int:
        return self.number_of_animals

    @property
    def single_animal(self) -> bool:
        return self.n_animals == 1

    @property
    def bkg_model(self) -> np.ndarray | None:
        if self.background_path.is_file():
            return cv2.imread(str(self.background_path))[..., 0]
        return None

    @bkg_model.setter
    def bkg_model(self, bkg: np.ndarray | None) -> None:
        if bkg is None:
            del self.bkg_model
            return
        # cv2.imwrite has given issues with paths containing chinese characters
        cv2.imencode(self.background_path.suffix, bkg)[1].tofile(self.background_path)
        logging.info(f"Background saved at {self.background_path}")

    @bkg_model.deleter
    def bkg_model(self) -> None:
        self.background_path.unlink(missing_ok=True)

    @property
    def ROI_mask(self) -> np.ndarray | None:
        if self.ROI_mask_path.is_file():
            return cv2.imread(str(self.ROI_mask_path))[..., 0]
        return None

    @ROI_mask.setter
    def ROI_mask(self, mask: np.ndarray | None) -> None:
        if mask is None:
            del self.ROI_mask
            return
        # cv2.imwrite has given issues with paths containing chinese characters
        cv2.imencode(self.ROI_mask_path.suffix, mask)[1].tofile(self.ROI_mask_path)
        logging.info(f"ROI mask saved at {self.ROI_mask_path}")

    @ROI_mask.deleter
    def ROI_mask(self) -> None:
        self.ROI_mask_path.unlink(missing_ok=True)

    @property
    def number_of_episodes(self) -> int:
        "Number of episodes in which the video is splitted for parallel processing"
        return len(self.episodes)

    # Paths and folders
    @property
    def preprocessing_folder(self) -> Path:
        return self.session_folder / "preprocessing"

    @property
    def background_path(self) -> Path:
        return self.preprocessing_folder / "background.png"

    @property
    def ROI_mask_path(self) -> Path:
        return self.preprocessing_folder / "ROI_mask.png"

    @property
    def trajectories_folder(self) -> Path:
        return self.session_folder / "trajectories"

    @property
    def crossings_detector_folder(self) -> Path:
        return self.session_folder / "crossings_detector"

    @property
    def individual_videos_folder(self) -> Path:
        return self.session_folder / "individual_videos"

    @property
    def accumulation_folder(self) -> Path:
        return self.session_folder / "accumulation"

    @property
    def id_images_folder(self) -> Path:
        return self.session_folder / "identification_images"

    @property
    def blobs_path(self) -> Path:
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been successfully performed"""
        return self.preprocessing_folder / "list_of_blobs.pickle"

    @property
    def idmatcher_results_path(self) -> Path:
        return self.session_folder / "matching_results"

    @property
    def global_fragments_path(self) -> Path:
        """get the path to save the list of global fragments after
        fragmentation"""
        return self.preprocessing_folder / "list_of_global_fragments.json"

    @property
    def fragments_path(self) -> Path:
        """get the path to save the list of global fragments after
        fragmentation"""
        return self.preprocessing_folder / "list_of_fragments.json"

    @property
    def path_to_session(self) -> Path:
        return self.session_folder / "session.json"

    @property
    def bbox_images_folder(self) -> Path:
        return self.session_folder / "bounding_box_images"

    @property
    def id_images_file_paths(self) -> list[Path]:
        # From version 6.0.0 id_images are saved using the .h5 extension, not .hdf5
        extension = "hdf5" if int(self.version.split(".")[0]) < 6 else "h5"
        try:
            return [
                self.id_images_folder / f"id_images_{e}.{extension}"
                for e in range(self.number_of_episodes)
            ]
        except AttributeError:
            # Loading a Session without the video files being present generates a session
            # without episodes. In this case, lets take all present files in id_images_folder
            paths: list[Path] = []
            for episode in count():
                path = self.id_images_folder / f"id_images_{episode}.{extension}"
                if not path.exists():
                    return paths
                paths.append(path)
            else:
                raise  # for PyLance

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {
            key: value
            for key, value in vars(cls).items()
            if not key.startswith("__")
            and not callable(value)
            and not callable(getattr(value, "__get__", None))
        }

    def update_paths(
        self, new_session_path: Path, user_video_paths_dir: Path | str | None = None
    ) -> None:
        """Update paths of objects (e.g. blobs_path, preprocessing_folder...)
        according to the location of the new Session object given
        by `new_session_path`.
        """
        logging.info(
            f"Searching video files: {[str(path.name) for path in self.video_paths]}"
        )
        folder_candidates: set[Path] = {
            self.video_paths[0],
            new_session_path,
            new_session_path.parent,
            self.session_folder.parent,
            self.session_folder,
            Path.cwd(),
        }
        if user_video_paths_dir is not None:
            folder_candidates.add(Path(user_video_paths_dir))

        need_to_save = False
        if self.session_folder != new_session_path:
            logging.info(
                f"Updated session folder from {self.session_folder} to"
                f" {new_session_path}"
            )
            self.session_folder = new_session_path
            need_to_save = True

        for folder_candidate in folder_candidates:
            if folder_candidate is None:
                continue

            try:
                # This can raise PermissionError if the path is not accessible
                if folder_candidate.is_file():
                    folder_candidate = folder_candidate.parent

                candidate_new_video_paths = [
                    folder_candidate / path.name for path in self.video_paths
                ]

                assert_all_files_exist(candidate_new_video_paths)
            except (FileNotFoundError, PermissionError):
                continue

            logging.info("All video files found in %s", folder_candidate)
            break
        else:
            raise FileNotFoundError(
                "Video file paths not found:\n    "
                + "\n    ".join(str(p) for p in self.video_paths)
            )

        if self.video_paths != candidate_new_video_paths:
            logging.info("Updating new video files paths")
            self.video_paths = candidate_new_video_paths
            need_to_save = True

        if need_to_save:
            self.save()

    @staticmethod
    def assert_video_paths(video_paths: Iterable[Path | str]) -> None:
        if not video_paths:
            raise IdtrackeraiError("Empty Video paths list")

        for path in video_paths:
            path = resolve_path(path)

            try:
                if path.is_dir():
                    raise IdtrackeraiError(
                        f'Directories ("{path}") are not allowed as video file paths'
                    )
            except PermissionError:
                raise IdtrackeraiError(f'Permission denied for the path "{path}"')

            if not path.is_file():
                raise IdtrackeraiError(f'Video file "{path}" not found')

            readable = cv2.VideoCapture(str(path)).grab()
            if not readable:
                raise IdtrackeraiError(f'Video file "{path}" not readable by OpenCV.')

    @staticmethod
    def get_info_from_video_paths(
        video_paths: Iterable[Path | str],
    ) -> tuple[int, int, float]:
        """Gets some information about the video from the video file itself.

        Returns:
            width: int, height: int, fps: float
        """

        widths, heights, fps = [], [], []
        for path in video_paths:
            cap = cv2.VideoCapture(str(path))
            widths.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            heights.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            try:
                fps.append(cap.get(cv2.CAP_PROP_FPS))
            except cv2.error:
                logging.warning(f"Cannot read frame per second for {path}")
                fps.append(None)
            cap.release()

        if len(set(widths)) != 1 or len(set(heights)) != 1:
            raise IdtrackeraiError("Video paths have different resolutions")

        if len(set(fps)) != 1:
            fps = [float(np.mean(fps))]
            logging.warning(
                f"Different frame rates detected ({fps}). "
                f"Setting the frame rate to the mean value: {fps[0]} fps"
            )

        return widths[0], heights[0], fps[0]

    @staticmethod
    def get_processing_episodes(
        video_paths: Sequence[Path | str],
        frames_per_episode: int,
        tracking_intervals=None,
    ) -> tuple[(int, list[int], list[list[int]], list[Episode])]:
        """Process the episodes by getting the number of frames in each video
        path and the tracking interval.

        Episodes are used to compute processes in parallel for different
        parts of the video. They are a tuple with

        - local start frame
        - local end frame
        - video path index
        - global start frame
        - global end frame

        where "local" means in the specific video path and "global" means in
        the whole (multi path) video

        Episodes are guaranteed to belong to a single video path and to have
        all of their frames (end not included) inside a the tracking interval
        """

        def in_which_interval(frame_number, intervals) -> int | None:
            for i, (start, end) in enumerate(intervals):
                if start <= frame_number < end:
                    return i
            return None

        for path in video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} not found")

        # total number of frames for every video path
        video_paths_n_frames = [
            int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
            for path in video_paths
        ]

        for n_frames, video_path in zip(video_paths_n_frames, video_paths):
            if n_frames <= 0:
                raise IdtrackeraiError(
                    f"OpenCV cannot read the number of frames in {video_path}"
                )
        number_of_frames = sum(video_paths_n_frames)

        # set full tracking interval if not defined
        if tracking_intervals is None:
            tracking_intervals = [[0, number_of_frames]]
        elif isinstance(tracking_intervals[0], int):
            tracking_intervals = [tracking_intervals]

        # find the global frames where the video path changes
        video_paths_changes = [0] + list(np.cumsum(video_paths_n_frames))

        # build an interval list like ("frame" refers to "global frame")
        #   [[first frame of video path 0, last frame of video path 0],
        #    [first frame of video path 1, last frame of video path 1],
        #    [...]]
        video_paths_intervals = list(pairwise(video_paths_changes))

        # find the frames where a tracking interval starts or ends
        tracking_intervals_changes = list(np.asarray(tracking_intervals).ravel())

        # Take into account tracking interval changes
        # and video path changes to compute episodes
        limits = video_paths_changes + tracking_intervals_changes

        # clean repeated limits and sort them
        limits = sorted(set(limits))

        # Create "long episodes" as the intervals between any video path
        # change or tracking interval change (keeping only the ones that
        # are inside a tracking interval)
        long_episodes = []
        for start, end in pairwise(limits):
            if (
                in_which_interval(start, tracking_intervals) is not None
            ) and 0 <= start < number_of_frames:
                long_episodes.append((start, end))

        # build definitive episodes by dividing long episodes to fit in
        # the FRAMES_PER_EPISODE restriction
        index = 0
        episodes = []
        for start, end in long_episodes:
            video_path_index = in_which_interval(start, video_paths_intervals)
            assert video_path_index is not None
            global_local_offset = video_paths_intervals[video_path_index][0]

            n_subepisodes = int((end - start) / (frames_per_episode + 1))
            new_episode_limits = np.linspace(start, end, n_subepisodes + 2, dtype=int)
            for new_start, new_end in pairwise(new_episode_limits):
                episodes.append(
                    Episode(
                        index=index,
                        local_start=new_start - global_local_offset,
                        local_end=new_end - global_local_offset,
                        video_path=resolve_path(video_paths[video_path_index]),
                        global_start=new_start,
                        global_end=new_end,
                    )
                )
                index += 1
        return number_of_frames, video_paths_n_frames, tracking_intervals, episodes

    @staticmethod
    def in_which_interval(frame_number, intervals) -> int | None:
        for i, (start, end) in enumerate(intervals):
            if start <= frame_number < end:
                return i
        return None

    @property
    def length_unit(self) -> float | None:
        """Length calibration factor for translating pixel units to user defined units. Property set in the Validator. Returns None if there are no calibrations."""
        if not hasattr(self, "length_calibrations"):
            return None

        values = []
        for c in self.length_calibrations:
            value = c.value()
            if value is not None:
                values.append(value)
        if not values:
            return None
        return fmean(values)

    def delete_data(self) -> None:
        """Deletes some folders with data, to make the outcome lighter.

        Which folders are deleted depends on the constant DATA_POLICY
        """
        logging.info(f"Data policy: {self.data_policy!r}")
        data_policy = self.data_policy.lower()

        if data_policy == "trajectories":
            remove_dir(self.bbox_images_folder)
            remove_dir(self.crossings_detector_folder)
            remove_dir(self.id_images_folder)
            remove_dir(self.accumulation_folder)
            remove_dir(self.preprocessing_folder)
        elif data_policy == "validation":
            remove_dir(self.bbox_images_folder)
            remove_dir(self.crossings_detector_folder)
            remove_dir(self.id_images_folder)
            remove_dir(self.accumulation_folder)
        elif data_policy == "knowledge_transfer":
            remove_dir(self.bbox_images_folder)
            remove_dir(self.crossings_detector_folder)
            remove_dir(self.id_images_folder)
        elif data_policy == "idmatcher.ai":
            remove_dir(self.bbox_images_folder)
            remove_dir(self.crossings_detector_folder)
        elif data_policy == "all":
            pass
        else:
            logging.error(
                "Data Policy is not valid. It has to be one of "
                f'{("trajectories", "validation", "knowledge_transfer", "idmatcher.ai", "all")}'
            )

    def compress_data(self) -> None:
        """Compress the identification images h5py files"""
        if not self.id_images_folder.exists():
            return

        tmp_path = self.session_folder / "tmp.h5py"

        for path in track(
            self.id_images_file_paths, "Compressing identification images"
        ):
            if not path.is_file():
                continue
            with (
                h5py.File(path, "r") as original_file,
                h5py.File(tmp_path, "w") as compressed_file,
            ):
                for key, data in original_file.items():
                    compressed_file.create_dataset(
                        key, data=data, compression="gzip" if "image" in key else None
                    )
            path.unlink()  # Windows needs this call before rename()
            tmp_path.rename(path)
