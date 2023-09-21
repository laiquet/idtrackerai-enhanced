import json
import logging
import sys
from importlib import metadata
from itertools import pairwise
from math import sqrt
from os import cpu_count
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import h5py
import numpy as np

from .utils import (
    CustomError,
    Episode,
    Timer,
    assert_all_files_exist,
    assert_knowledge_transfer_is_possible,
    build_ROI_mask_from_list,
    create_dir,
    json_default,
    json_object_hook,
    remove_dir,
    remove_file,
    resolve_path,
    track,
)


class Video:
    """
    A class containing the main features of the video.

    This class includes properties of the video by itself, user defined
    parameters for the tracking, and other properties that are generated
    throughout the tracking process.

    We use this class as a storage of data coming from different processes.
    However, this is bad practice and it will change in the future.
    """

    velocity_threshold: float
    erosion_kernel_size: int
    ratio_accumulated_images: float
    accumulation_folder: Path
    # FIXME it should depend on self.session_folder
    # return self.session_folder / f"accumulation_{self.accumulation_trial}"
    individual_fragments_stats: dict
    percentage_of_accumulated_images: list[float]
    # TODO: move to accumulation_manager.py
    # TODO: move to accumulation_manager.py
    session_folder: Path
    # TODO remove these defaults, they are already in __main__
    setup_points: dict[str, list[tuple[int, int]]]

    median_body_length: float
    """median of the diagonals of individual blob's bounding boxes"""

    # TODO: move tracker.py
    first_frame_first_global_fragment: list

    # During validation (in validation GUI)
    identities_groups: dict
    """Groups of identities stored during the validation of the tracking
    in the validation GUI. This is useful to group identities in different
    classes depending on the experiment.

    This feature was coded because some users require indicating classes
    of individuals but we do not use it in the lab."""
    episodes: list[Episode]
    """Indicates the starting and ending frames of each video episode.
    Video episodes are used for parallelization of some processes"""

    original_width: int
    """Original video width in pixels. It does not consider the resolution
    reduction factor defined by the user"""
    original_height: int
    """Original video width in pixels. It does not consider the resolution
    reduction factor defined by the user"""
    frames_per_second: int
    """Video frame rate in frames per second obtained by OpenCV from the
    video file"""
    accumulation_statistics: dict[str, list]
    accumulation_statistics_data: list[dict[str, list]]
    number_of_error_frames: int = -1
    """The number of frames with more blobs than animals. Set on animals_detection."""
    estimated_accuracy: float | None = None
    accumulation_trial: int = 0
    identities_labels: list[str] | None = None
    """A list with a name for every identity. Defined and used in validator"""
    background_from_segmentation_gui: np.ndarray | None = None
    """Background set by segmentation app to save when the app closes"""

    video_paths: list[Path] = []
    """List of paths to the different files the video is composed of.
    If the video is a single file, the list will have length 1"""
    number_of_animals: int = 0
    intensity_ths: None | Sequence[int] = None
    area_ths: None | Sequence[int] = None
    # bkg_model: None | np.ndarray = None
    session: str = "no_name"
    output_dir: Path | None | str = None
    tracking_intervals: list | None = None
    resolution_reduction: float = 1.0
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
    ] = "all"
    id_image_size: list[int] = []
    """ Shape of the Blob's identification images (width, height, n_channels)"""
    protocol3_action: Literal["ask", "abort", "continue"] = "ask"
    convert_trajectories_to_csv_and_json: bool = True
    add_time_column_to_csv: bool = False
    """Add a time column (in seconds) to csv trajectory filesy"""
    version: str
    """Version of idtracker.ai"""

    def set_parameters(self, **parameters):
        """Sets parameters to self only if they are present in the class annotations.
        The set of non recognized parameters names is returned"""
        non_recognized_parameters: set[str] = set()
        for param, value in parameters.items():
            lower_param = param.lower()
            if lower_param in self.__class__.__annotations__:
                setattr(self, lower_param, value)
            else:
                non_recognized_parameters.add(param)
        return non_recognized_parameters

    def prepare_tracking(self):
        """Initializes the video object, checking all parameters"""
        logging.debug("Initializing Video object")
        self.version = metadata.version("idtrackerai")

        if not isinstance(self.video_paths, list):
            video_paths = [self.video_paths]
        else:
            video_paths = self.video_paths
        self.assert_video_paths(video_paths)
        self.video_paths = [Path(path).expanduser().resolve() for path in video_paths]
        logging.info(
            "Setting video_paths to:\n    " + "\n    ".join(map(str, self.video_paths))
        )

        if self.area_ths is None:
            raise CustomError("Missing area thresholds parameter")

        if self.intensity_ths is None:
            raise CustomError("Missing intensity thresholds parameter")

        self.accumulation_statistics_data = []

        if self.knowledge_transfer_folder is not None:
            self.knowledge_transfer_folder = resolve_path(
                self.knowledge_transfer_folder
            )
            if not self.knowledge_transfer_folder.exists():
                raise CustomError(
                    f'Knowledge transfer folder "{self.knowledge_transfer_folder}" not'
                    " found"
                )

        self.original_width, self.original_height, self.frames_per_second = (
            self.get_info_from_video_paths(self.video_paths)
        )
        self.number_of_frames, _, self.tracking_intervals, self.episodes = (
            self.get_processing_episodes(
                self.video_paths, self.frames_per_episode, self.tracking_intervals
            )
        )

        logging.info(
            f"The video has {self.number_of_frames} "
            f"frames ({self.number_of_episodes} episodes)"
        )
        if len(self.episodes) < 10:
            for e in self.episodes:
                video_name = self.video_paths[e.video_path_index].name
                logging.info(
                    f"\tEpisode {e.index}, frames ({e.local_start} "
                    f"=> {e.local_end}) of /{video_name}"
                )
        assert self.number_of_episodes > 0

        if self.output_dir is not None:
            self.session_folder = (
                resolve_path(self.output_dir) / f"session_{self.session.strip()}"
            )
        else:
            self.session_folder = (
                self.video_folder / f"session_{self.session.strip()}"
            ).resolve()
        create_dir(self.session_folder)
        create_dir(self.preprocessing_folder)

        self.ROI_mask = build_ROI_mask_from_list(
            self.roi_list,
            self.resolution_reduction,
            self.original_width,
            self.original_height,
        )

        if isinstance(self.id_image_size, int):
            self.id_image_size = [self.id_image_size, self.id_image_size, 1]
        else:
            self.id_image_size = []

        if self.knowledge_transfer_folder is not None:
            self.id_image_size = assert_knowledge_transfer_is_possible(
                self.knowledge_transfer_folder, self.n_animals
            )

        if self.number_of_parallel_workers <= 0:
            computer_CPUs = cpu_count()
            if computer_CPUs is not None:
                if self.number_of_parallel_workers == 0:
                    self.number_of_parallel_workers = (computer_CPUs + 1) // 2
                elif self.number_of_parallel_workers < 0:
                    self.number_of_parallel_workers += computer_CPUs
        logging.info("Number of parallel jobs: %d", self.number_of_parallel_workers)

        if self.number_of_animals == 0 and not self.track_wo_identities:
            raise CustomError(
                "Cannot track with an undefined number of animals (n_animals = 0)"
                " when tracking with identities"
            )

        self.bkg_model = self.background_from_segmentation_gui  # has a setter
        self.__dict__.pop("background_from_segmentation_gui", None)

        self.first_frame_first_global_fragment = []
        self.identities_groups = {}
        self.setup_points = {}

        # Processes timers
        self.general_timer = Timer("Tracking session")
        self.detect_animals_timer = Timer("Animal detection")
        self.crossing_detector_timer = Timer("Crossing detection")
        self.fragmentation_timer = Timer("Fragmentation")
        self.tracking_timer = Timer("Tracking")
        self.protocol1_timer = Timer("Protocol 1")
        self.protocol2_timer = Timer("Protocol 2")
        self.protocol3_pretraining_timer = Timer("Protocol 3 pre-training")
        self.protocol3_accumulation_timer = Timer("Protocol 3 accumulation")
        self.identify_timer = Timer("Identification")
        self.impossible_jumps_timer = Timer("Impossible jumps correction")
        self.crossing_solver_timer = Timer("Crossings solver")
        self.create_trajectories_timer = Timer("Trajectories creation")

        self.general_timer.start()

    def __str__(self) -> str:
        return f"<session {self.session_folder}>"

    def set_id_image_size(self, median_body_length: int | float, reset=False):
        self.median_body_length = median_body_length
        if reset or not self.id_image_size:
            side_length = int(median_body_length / sqrt(2))
            side_length += side_length % 2
            self.id_image_size = [side_length, side_length, 1]
        logging.info(f"Identification image size set to {self.id_image_size}")

    @property
    def n_animals(self):
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
    def bkg_model(self, bkg: np.ndarray | None):
        if bkg is None:
            del self.bkg_model
        else:
            cv2.imwrite(str(self.background_path), bkg)
            logging.info(f"Background saved at {self.background_path}")

    @bkg_model.deleter
    def bkg_model(self):
        self.background_path.unlink(missing_ok=True)

    @property
    def ROI_list(self):
        """Fixes compatibility issues"""
        return self.roi_list

    @property
    def ROI_mask(self) -> np.ndarray | None:
        if self.ROI_mask_path.is_file():
            return cv2.imread(str(self.ROI_mask_path))[..., 0]
        return None

    @ROI_mask.setter
    def ROI_mask(self, mask: np.ndarray | None):
        if mask is None:
            del self.ROI_mask
        else:
            cv2.imwrite(str(self.ROI_mask_path), mask)
            logging.info(f"ROI mask saved at {self.ROI_mask_path}")

    @ROI_mask.deleter
    def ROI_mask(self):
        self.ROI_mask_path.unlink(missing_ok=True)

    @property
    def video_folder(self) -> Path:
        """Directory where video was stored. Parent of video_path.

        Returns
        -------
        str
            Path to the video folder where the video to be tracked was stored.
        """
        return self.video_paths[0].parent

    @property
    def number_of_episodes(self):
        """Number of episodes in which the video is splitted for parallel
        processing.

        Returns
        -------
        int
            Number of parts in which the videos is splitted.

        See Also
        --------
        :int:`~idtrackerai.constants.FRAMES_PER_EPISODE`
        """
        return len(self.episodes)

    @property
    def width(self):
        """Video width in pixels after applying the resolution reduction
        factor.

        Returns
        -------
        int
            Video width in pixels after applying the resolution reduction
            factor defined by the user.
        """
        return int(self.original_width * self.resolution_reduction + 0.5)

    @property
    def height(self):
        """Video height in pixels after applying the resolution reduction
        factor.

        Returns
        -------
        int
            Video height in pixels after applying the resolution reduction
            factor.
        """
        return int(self.original_height * self.resolution_reduction + 0.5)

    # TODO: move to crossings_detection.py
    @property
    def median_body_length_full_resolution(self):
        """Median body length in pixels in full frame resolution
        (i.e. without considering the resolution reduction factor)
        """
        return self.median_body_length / self.resolution_reduction

    # Paths and folders
    # TODO: The different processes should create and store the path to the
    # folder where they save the data
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
    def pretraining_folder(self) -> Path:
        return self.session_folder / "pretraining"

    @property
    def individual_videos_folder(self) -> Path:
        return self.session_folder / "individual_videos"

    @property
    def auto_accumulation_folder(self) -> Path:
        return self.session_folder / f"accumulation_{self.accumulation_trial}"

    @property
    def id_images_folder(self) -> Path:
        return self.session_folder / "identification_images"

    # TODO: This should probably be the only path that should be stored in
    # Video.

    @property
    def blobs_path(self) -> Path:
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been successfully performed"""
        return self.preprocessing_folder / "list_of_blobs.pickle"

    @property
    def blobs_no_gaps_path(self) -> Path:
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been successfully performed"""
        return self.preprocessing_folder / "list_of_blobs_no_gaps.pickle"

    @property
    def blobs_path_validated(self) -> Path:
        return self.preprocessing_folder / "list_of_blobs_validated.pickle"

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
    def path_to_video_object(self) -> Path:
        return self.session_folder / "video_object.json"

    @property
    def ground_truth_path(self) -> Path:
        return self.video_folder / "_groundtruth.npy"

    @property
    def segmentation_data_folder(self) -> Path:
        return self.session_folder / "segmentation_data"

    @property
    def id_images_file_paths(self) -> list[Path]:
        return [
            self.id_images_folder / f"id_images_{e}.hdf5"
            for e in range(self.number_of_episodes)
        ]

    @classmethod
    def defaults(cls):
        return {
            key: value
            for key, value in vars(cls).items()
            if not key.startswith("__")
            and not callable(value)
            and not callable(getattr(value, "__get__", None))
        }

    # Methods
    def save(self):
        """Saves the instantiated Video object"""
        logging.info(f"Saving video object in {self.path_to_video_object}")
        dict_to_save = (self.defaults() | vars(self)).copy()
        dict_to_save.pop("episodes", None)
        self.path_to_video_object.write_text(
            json.dumps(dict_to_save, default=json_default, indent=4)
        )

    @classmethod
    def load(cls, path: Path | str, video_paths_dir: Path | None = None) -> "Video":
        """Load a video object stored in a JSON file"""
        path = resolve_path(path)
        logging.info(f"Loading Video from {path}")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        if not path.is_file():
            path /= "video_object.json"
            if not path.is_file():
                path = path.with_suffix(".npy")
                if not path.is_file():
                    raise FileNotFoundError(f"{path} not found")

        if path.suffix == ".npy":
            video_dict = cls.open_from_v4(path)
        else:
            with open(path, "r", encoding="utf_8") as file:
                video_dict = json.load(file, object_hook=json_object_hook)

        if "n_animals" not in video_dict and "number_of_animals" in video_dict:
            video_dict["n_animals"] = video_dict["number_of_animals"]

        video = cls.__new__(cls)
        video.__dict__.update(video_dict)
        video.update_paths(path.parent, video_paths_dir)
        try:
            _, _, _, video.episodes = video.get_processing_episodes(
                video.video_paths, video.frames_per_episode, video.tracking_intervals
            )
        except AttributeError:
            logging.warning(
                "Could not load video episodes probably due to loading an old version"
                " session"
            )
        return video

    @classmethod
    def open_from_v4(cls, path: Path) -> dict:
        from . import network

        logging.warning("Loading from v4: %s", path)

        # v4 compatibility
        sys.modules["idtrackerai.tracker.network.network_params"] = network
        _dict: dict = np.load(path, allow_pickle=True).item().__dict__
        del sys.modules["idtrackerai.tracker.network.network_params"]

        _dict["version"] = "4.0.12 or below"
        _dict["video_paths"] = list(map(Path, _dict.pop("_video_paths")))
        _dict["session_folder"] = path.parent
        _dict["median_body_length"] = _dict.pop("_median_body_length")
        _dict["frames_per_second"] = _dict.pop("_frames_per_second")
        _dict["original_width"] = _dict.pop("_original_width")
        _dict["original_height"] = _dict.pop("_original_height")
        _dict["number_of_frames"] = _dict.pop("_number_of_frames")
        _dict["identities_groups"] = _dict.pop("_identities_groups")
        _dict["id_image_size"] = list(_dict.pop("_identification_image_size"))
        _dict["setup_points"] = _dict.pop("_setup_points")
        _dict["number_of_animals"] = _dict["_user_defined_parameters"][
            "number_of_animals"
        ]
        _dict["tracking_intervals"] = _dict["_user_defined_parameters"][
            "tracking_interval"
        ]
        _dict["resolution_reduction"] = _dict["_user_defined_parameters"][
            "resolution_reduction"
        ]
        _dict["track_wo_identities"] = _dict["_user_defined_parameters"][
            "track_wo_identification"
        ]
        _dict["accumulation_folder"] = (
            path.parent / Path(_dict.pop("_accumulation_folder")).name
        )
        _dict["_user_defined_parameters"].pop("mask")
        return _dict

    def update_paths(
        self, new_video_object_path: Path, user_video_paths_dir: Path | None = None
    ):
        """Update paths of objects (e.g. blobs_path, preprocessing_folder...)
        according to the new location of the new video object given
        by `new_video_object_path`.

        Parameters
        ----------
        new_video_object_path : str
            Path to a video_object.npy
        """
        logging.info(
            f"Searching video files: {[str(path.name) for path in self.video_paths]}"
        )
        folder_candidates: set[Path | None] = {
            user_video_paths_dir,
            self.video_paths[0],
            new_video_object_path,
            new_video_object_path.parent,
            self.session_folder.parent,
            self.session_folder,
        }

        for folder_candidate in folder_candidates:
            if folder_candidate is None:
                continue
            if folder_candidate.is_file():
                folder_candidate = folder_candidate.parent

            candidate_new_video_paths = [
                folder_candidate / path.name for path in self.video_paths
            ]

            try:
                assert_all_files_exist(candidate_new_video_paths)
            except FileNotFoundError:
                continue

            logging.info(f"All video files found on {folder_candidate}")
            found = True
            break
        else:
            found = False
            candidate_new_video_paths = []
            logging.error(f"Video file paths not found: {self.video_paths}")

        need_to_save = False
        if self.session_folder != new_video_object_path:
            logging.info(
                f"Updated session folder from {self.session_folder} to"
                f" {new_video_object_path}"
            )
            self.session_folder = new_video_object_path
            need_to_save = True

        if found and self.video_paths != candidate_new_video_paths:
            logging.info("Updating new video files paths")
            self.video_paths = candidate_new_video_paths
            need_to_save = True

        if need_to_save:
            self.save()

    @staticmethod
    def assert_video_paths(video_paths: Iterable[Path | str]):
        if not video_paths:
            raise CustomError("Empty Video paths list")

        for path in video_paths:
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise CustomError(f'Video file "{path}" not found')

            readable = cv2.VideoCapture(str(path)).grab()
            if not readable:
                raise CustomError(f'Video file "{path}" not readable by OpenCV.')

    @staticmethod
    def get_info_from_video_paths(video_paths: Iterable[Path | str]):
        """Gets some information about the video from the video file itself.

        Returns:
            width: int, height: int, fps: int
        """

        widths, heights, fps = [], [], []
        for path in video_paths:
            cap = cv2.VideoCapture(str(path))
            widths.append(int(cap.get(3)))
            heights.append(int(cap.get(4)))

            try:
                fps.append(int(cap.get(5)))
            except cv2.error:
                logging.warning(f"Cannot read frame per second for {path}")
                fps.append(None)
            cap.release()

        if len(set(widths)) != 1 or len(set(heights)) != 1:
            raise CustomError("Video paths have different resolutions")

        if len(set(fps)) != 1:
            fps = [int(np.mean(fps))]
            logging.warning(
                f"Different frame rates detected ({fps}). "
                f"Setting the frame rate to the mean value: {fps[0]} fps"
            )

        return widths[0], heights[0], fps[0]

    # Methods to create folders where to store data
    # TODO: Some of these methods should go to the classes corresponding to
    # the process.

    def create_accumulation_folder(self, iteration_number=None, delete=False):
        """Folder in which the model generated while accumulating is stored
        (after pretraining)
        """
        if iteration_number is None:
            iteration_number = self.accumulation_trial
        self.accumulation_folder = (
            self.session_folder / f"accumulation_{iteration_number}"
        )
        # FIXME
        create_dir(self.accumulation_folder, remove_existing=delete)

    # Some methods related to the accumulation process
    # TODO: Move to accumulation_manager.py
    def init_accumulation_statistics_attributes(self):
        self.accumulation_statistics = {
            "n_accumulated_global_fragments": [],
            "n_non_certain_global_fragments": [],
            "n_randomly_assigned_global_fragments": [],
            "n_nonconsistent_global_fragments": [],
            "n_nonunique_global_fragments": [],
            "n_acceptable_global_fragments": [],
            "ratio_of_accumulated_images": [],
        }

    @staticmethod
    def get_processing_episodes(
        video_paths, frames_per_episode, tracking_intervals=None
    ) -> tuple[(int, list[int], list[list[int]], list[Episode])]:
        """Process the episodes by getting the number of frames in each video
        path and the tracking interval.

        Episodes are used to compute processes in parallel for different
        parts of the video. They are a tuple with
            (local start frame,
            local end frame,
            video path index,
            global start frame,
            global end frame)
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

        # total number of frames for every video path
        video_paths_n_frames = [
            int(cv2.VideoCapture(str(path)).get(7)) for path in video_paths
        ]

        for n_frames, video_path in zip(video_paths_n_frames, video_paths):
            if n_frames <= 0:
                raise CustomError(
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
        tracking_intervals_changes = list(np.asarray(tracking_intervals).flatten())

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
                        video_path_index=video_path_index,
                        global_start=new_start,
                        global_end=new_end,
                    )
                )
                index += 1
        return number_of_frames, video_paths_n_frames, tracking_intervals, episodes

    @staticmethod
    def in_which_interval(frame_number, intervals):
        for i, (start, end) in enumerate(intervals):
            if start <= frame_number < end:
                return i
        return None

    def delete_data(self):
        """Deletes some folders with data, to make the outcome lighter.

        Which folders are deleted depends on the constant DATA_POLICY
        """

        logging.info(f'Data policy: "{self.data_policy}"')

        if self.data_policy == "trajectories":
            remove_dir(self.segmentation_data_folder)
            remove_file(self.global_fragments_path)
            remove_dir(self.crossings_detector_folder)
            remove_dir(self.id_images_folder)
            for path in self.session_folder.glob("accumulation_*"):
                remove_dir(path)
            remove_dir(self.session_folder / "pretraining")
            remove_dir(self.preprocessing_folder)
        elif self.data_policy == "validation":
            remove_dir(self.segmentation_data_folder)
            remove_file(self.global_fragments_path)
            remove_dir(self.crossings_detector_folder)
            remove_dir(self.id_images_folder)
            for path in self.session_folder.glob("accumulation_*"):
                remove_dir(path)
            remove_dir(self.session_folder / "pretraining")
        elif self.data_policy == "knowledge_transfer":
            remove_dir(self.segmentation_data_folder)
            remove_file(self.global_fragments_path)
            remove_dir(self.crossings_detector_folder)
            remove_dir(self.id_images_folder)
        elif self.data_policy == "idmatcher.ai":
            remove_dir(self.segmentation_data_folder)
            remove_file(self.global_fragments_path)
            remove_dir(self.crossings_detector_folder)

    def compress_data(self):
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
