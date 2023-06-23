# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)
import json
import logging
import sys
from copy import copy
from importlib import metadata
from math import sqrt
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np

from .utils import (
    Episode,
    Timer,
    assert_all_files_exist,
    build_ROI_mask_from_list,
    check_if_identity_transfer_is_possible,
    conf,
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

    accumulation_step: int
    velocity_threshold: float
    erosion_kernel_size: int
    ratio_accumulated_images: float
    accumulation_folder: Path
    # FIXME it should depend on self.session_folder
    # return self.session_folder / f"accumulation_{self.accumulation_trial}"
    individual_fragments_stats: dict
    estimated_accuracy: float | None = None
    percentage_of_accumulated_images: list[float]
    # TODO: move to accumulation_manager.py
    accumulation_trial: int = 0
    # TODO: move to accumulation_manager.py
    session_folder: Path
    # TODO remove these defaults, they are already in __main__

    id_image_size: list[int]
    """ Shape of the Blob's identification images (width, height, n_channels)"""

    episodes: list[Episode]
    """Indicates the starting and ending frames of each video episode.
    Video episodes are used for parallelization of some processes"""

    video_paths: list[Path]
    """List of paths to the different files the video is composed of.
    If the video is a single file, the list will have length 1"""

    original_width: int
    """Original video width in pixels. It does not consider the resolution
    reduction factor defined by the user"""

    original_height: int
    """Original video width in pixels. It does not consider the resolution
    reduction factor defined by the user"""

    frames_per_second: int
    """Video frame rate in frames per second obtained by OpenCV from the
    video file"""

    number_of_error_frames: int = -1
    """The number of frames with more blobs than animals. Set on animals_detection."""

    accumulation_statistics: dict[str, list]

    accumulation_statistics_data: list[dict[str, list]]

    identities_labels: list[str] | None = None
    """A list with a name for every identity. Defined and used in validator"""

    def __init__(
        self,
        video_paths: list[Path | str],
        number_of_animals,
        intensity_ths,
        area_ths,
        output_dir: Path | None,
        session,
        tracking_intervals: list | None,
        resolution_reduction: float,
        roi_list: list[str] | None,
        use_bkg: bool,
        track_wo_identities: bool,
        check_segmentation: bool,
        identity_transfer: bool,
        knowledge_transfer_folder: Path | None,
        bkg_model,
        **kwargs,
    ):
        """Initializes a video object

        Parameters
        ----------
        video_path : str
            Path to a video file
        """
        if kwargs:
            logging.info(
                f"Ignoring the next arguments in Video.__init__():\n{kwargs.keys()}"
            )

        logging.debug("Initializing Video object")
        self.use_bkg = use_bkg
        self.check_segmentation = check_segmentation
        self.track_wo_identities = track_wo_identities
        """Flag indication the tracking will be performed without identities"""
        self.intensity_ths = intensity_ths
        self.area_ths = area_ths
        self.knowledge_transfer_folder = knowledge_transfer_folder
        self.resolution_reduction = resolution_reduction
        self.number_of_animals = int(number_of_animals)
        """Number of animals in the video indicated by user"""
        self.set_video_paths(video_paths)
        self.data_policy: str = conf.DATA_POLICY
        self.frames_per_episode: int = conf.frames_per_episode
        self.version = metadata.version("idtrackerai")
        self.protocol3_action: str = conf.protocol3_action
        self.accumulation_statistics_data = [None] * (
            conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS + 1
        )

        if self.knowledge_transfer_folder:
            self.knowledge_transfer_folder = Path(
                self.knowledge_transfer_folder
            ).resolve()
            assert (
                self.knowledge_transfer_folder.exists()
            ), f"{self.knowledge_transfer_folder} not found"

        self.original_width, self.original_height, self.frames_per_second = (
            self.get_info_from_video_paths(self.video_paths)
        )
        self.number_of_frames, _, self.tracking_intervals, self.episodes = (
            self.get_processing_episodes(
                self.video_paths, self.frames_per_episode, tracking_intervals
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

        if output_dir is not None:
            self.session_folder = (output_dir / f"session_{session.strip()}").resolve()
        else:
            self.session_folder = (
                self.video_folder / f"session_{session.strip()}"
            ).resolve()
        create_dir(self.session_folder)
        create_dir(self.preprocessing_folder)

        self.ROI_list = roi_list

        self.ROI_mask = build_ROI_mask_from_list(
            roi_list, resolution_reduction, self.original_width, self.original_height
        )

        if conf.IDENTIFICATION_IMAGE_SIZE > 0:
            self.id_image_size = [
                conf.IDENTIFICATION_IMAGE_SIZE,
                conf.IDENTIFICATION_IMAGE_SIZE,
                1,
            ]
        else:
            self.id_image_size = []

        if identity_transfer:
            # TODO: the id_image_size is not really passed by
            # the used but inferred from the knowledge transfer folder
            self.identity_transfer, self.id_image_size = (
                check_if_identity_transfer_is_possible(
                    self.number_of_animals, self.knowledge_transfer_folder
                )
            )
        else:
            self.identity_transfer = False

        self.bkg_model = bkg_model  # has a setter

        # Attributes computed by other processes in the tracking
        # During crossing detection
        self.median_body_length: float
        """median of the diagonals of individual blob's bounding boxes"""

        # TODO: move tracker.py
        self.first_frame_first_global_fragment = []  # updated later

        # During validation (in validation GUI)
        self.identities_groups = {}
        """Groups of identities stored during the validation of the tracking
        in the validation GUI. This is useful to group identities in different
        classes depending on the experiment.

        This feature was coded because some users require indicating classes
        of individuals but we do not use it in the lab."""

        self.setup_points: dict[str, list[tuple[float, float]]] = {}
        """Setup points"""

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
    def single_animal(self) -> bool:
        return self.number_of_animals == 1

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
    def ROI_mask(self) -> np.ndarray | None:
        if self.ROI_mask_path.is_file():
            return cv2.imread(str(self.ROI_mask_path))[..., 0].astype(bool)
        return None

    @ROI_mask.setter
    def ROI_mask(self, mask: np.ndarray | None):
        if mask is None:
            del self.ROI_mask
        else:
            cv2.imwrite(str(self.ROI_mask_path), (mask * 255).astype(np.uint8))
            logging.info(f"ROI mask saved at {self.ROI_mask_path}")

    @ROI_mask.deleter
    def ROI_mask(self):
        self.ROI_mask_path.unlink(missing_ok=True)

    def set_video_paths(self, video_paths: list[Path | str]):
        if not isinstance(video_paths, list):
            video_paths = [video_paths]
        self.assert_video_paths(video_paths)
        self.video_paths = [Path(path).expanduser().resolve() for path in video_paths]
        to_print = "Setting video_paths to:\n    " + "\n    ".join(
            map(str, self.video_paths)
        )
        logging.info(to_print)

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

    # Methods
    def save(self):
        """Saves the instantiated Video object"""
        logging.info(f"Saving video object in {self.path_to_video_object}")
        dict_to_save = copy(self.__dict__)
        dict_to_save.pop("episodes", None)
        dict_to_save.pop("_model_area", None)
        dict_to_save.pop("_accumulation_network_params", None)
        self.path_to_video_object.write_text(
            json.dumps(dict_to_save, default=json_default, indent=4)
        )

    @classmethod
    def load(cls, path: Path | str, video_paths_dir: Path | None = None) -> "Video":
        """Load a video object stored in a JSON file"""
        path = resolve_path(path)
        logging.info(f"Loading Video from {path}")
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
    def assert_video_paths(
        video_paths: Iterable[Path | str], accepted_extensions: list[str] | None = None
    ):
        accepted_extensions = accepted_extensions or conf.AVAILABLE_VIDEO_EXTENSION
        assert video_paths, "Empty video_paths list"

        for path in video_paths:
            path = Path(path).expanduser().resolve()
            assert path.is_file(), f"Video file {path} not found"
            assert (
                path.suffix in accepted_extensions
            ), f"Supported video extensions are {accepted_extensions}"

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

        assert len(set(widths)) == 1, "Video paths have different resolutions"
        assert len(set(heights)) == 1, "Video paths have different resolutions"
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
        video_paths_intervals = list(
            zip(video_paths_changes[:-1], video_paths_changes[1:])
        )

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
        for start, end in zip(limits[:-1], limits[1:]):
            if (
                in_which_interval(start, tracking_intervals) is not None
            ) and 0 <= start < number_of_frames:
                long_episodes.append((start, end))

        # build definitive episodes by dividing long episodes to fit in
        # the conf.FRAMES_PER_EPISODE restriction
        index = 0
        episodes = []
        for start, end in long_episodes:
            video_path_index = in_which_interval(start, video_paths_intervals)
            assert video_path_index is not None
            global_local_offset = video_paths_intervals[video_path_index][0]

            n_subepisodes = int((end - start) / (frames_per_episode + 1))
            new_episode_limits = np.linspace(start, end, n_subepisodes + 2, dtype=int)
            for new_start, new_end in zip(
                new_episode_limits[:-1], new_episode_limits[1:]
            ):
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

        logging.info(f"Data policy: {self.data_policy}")

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
