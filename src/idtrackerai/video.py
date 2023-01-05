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
from math import sqrt
from pathlib import Path

import cv2
import numpy as np

from idtrackerai.utils import (
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
    def __init__(
        self,
        video_paths,
        number_of_animals,
        intensity_ths,
        area_ths,
        output_dir: Path | None = None,
        session="no_name",
        tracking_intervals=None,
        resolution_reduction=1,
        ROI_list: list | None = None,
        ROI_mask: np.ndarray | None = None,
        use_bkg: bool = False,
        bkg_model=None,
        setup_points=None,
        track_wo_identities: bool = False,
        sigma_gaussian_blurring=None,
        check_segmentation=False,
        identity_transfer=False,
        knowledge_transfer_folder: Path | None = None,
        **kwargs,
    ):
        """Initializes a video object

        Parameters
        ----------
        video_path : str
            Path to a video file
        """
        if sigma_gaussian_blurring is None:
            sigma_gaussian_blurring = conf.SIGMA_GAUSSIAN_BLURRING
        if kwargs:
            logging.info(
                f"Ignoring the next arguments in Video.__init__():\n{kwargs.keys()}"
            )

        logging.debug("Initializing Video object")
        self.use_bkg = use_bkg
        self.check_segmentation = check_segmentation
        self.setup_points = setup_points
        self.track_wo_identities = track_wo_identities
        """Flag indication the tracking will be performed without identities"""
        self.intensity_ths = intensity_ths
        self.area_ths = area_ths
        self.knowledge_transfer_folder = (
            knowledge_transfer_folder
            if knowledge_transfer_folder
            else conf.knowledge_transfer_folder
        )
        self.resolution_reduction = resolution_reduction
        self.number_of_animals = int(number_of_animals)
        self.video_paths = video_paths  # has a setter
        self.tracking_intervals = tracking_intervals
        self.sigma_gaussian_blurring = sigma_gaussian_blurring

        if self.knowledge_transfer_folder:
            self.knowledge_transfer_folder = Path(
                self.knowledge_transfer_folder
            ).resolve()
            assert (
                self.knowledge_transfer_folder.exists()
            ), f"{self.knowledge_transfer_folder} not found"

        (
            self._original_width,
            self._original_height,
            self._frames_per_second,
        ) = self.get_info_from_video_paths(self.video_paths)
        (
            self.number_of_frames,
            _,
            self.tracking_intervals,
            self._episodes,
        ) = self.get_processing_episodes(self.video_paths, self.tracking_intervals)

        logging.info(f"The video has {self.number_of_frames} frames")
        logging.info(f"The video has {self.number_of_episodes} episodes:")
        for e in self.episodes:
            video_name = self.video_paths[e.video_path_index].name
            logging.info(
                f"\tEpisode {e.index}, frames ({e.local_start} => {e.local_end}) of /{video_name}"
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

        if ROI_mask is not None:
            self.ROI_mask = ROI_mask
        elif ROI_list is not None:
            self.ROI_mask = build_ROI_mask_from_list(
                self.original_width, self.original_height, list_of_ROIs=ROI_list
            )
        else:
            self.ROI_mask = None

        self.id_image_size: list[int] = []
        """ Shape of the Blob's identification images
        (width, height, n_channels)"""

        if identity_transfer:
            # TODO: the id_image_size is not really passed by
            # the used but inferred from the knowledge transfer folder
            (
                self.identity_transfer,
                self.id_image_size,
            ) = check_if_identity_transfer_is_possible(
                self.number_of_animals, self.knowledge_transfer_folder
            )
        else:
            self.identity_transfer = False

        self.bkg_model = bkg_model  # has a setter

        # TODO: HARDCODED _number_of_channels. Change if color information is used.
        # Currently idtracker.ai does not rely on color. All color videos
        # are converted to gray scale so the number of channels is forced to
        # always be one. This should be changed if color images are used for
        # identification, as this attributed is used to created the
        # identification images.
        self.number_of_channels: int = 1
        """Number of channels in the video"""

        # Attributes computed by other processes in the tracking
        # During crossing detection
        self.median_body_length: float | int
        """median of the diagonals of individual blob's bounding boxes"""
        self.there_are_crossings: bool
        # During tracking (protocol cascade)
        self._identity_transfer = None  # updated later
        self._tracking_with_knowledge_transfer = False  # updated later
        self._first_frame_first_global_fragment = []  # updated later

        # During validation (in validation GUI)
        self._identities_groups = {}  # updated later
        # self.accumulation_iteration = 0

        # Flag to decide which type of interpolation is done. This flag
        # is updated when we update a blob centroid
        self._is_centroid_updated = False

        # Processes states

        self._has_residual_identification = False  # residual identification
        self._has_impossible_jumps_solved = False  # post-processing

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

    def set_id_image_size(self, median_body_length: int | float, reset=False):
        self.median_body_length = median_body_length
        if reset or not self.id_image_size:
            side_length = int(median_body_length / sqrt(2))
            side_length += side_length % 2
            self.id_image_size = [side_length, side_length, self.number_of_channels]
        logging.info(f"Identification image size set to {self.id_image_size}")

    @property
    def single_animal(self) -> bool:
        return self.number_of_animals == 1

    @property
    def bkg_model(self) -> np.ndarray | None:
        if self.background_path.is_file():
            return (
                cv2.imread(str(self.background_path))[..., 0].astype(np.float32)
                * self.bkg_norm
            )

        else:
            return None

    @bkg_model.setter
    def bkg_model(self, bkg: np.ndarray | None):
        if bkg is None:
            del self.bkg_model
        else:
            self.bkg_norm = bkg.max() / 255
            cv2.imwrite(
                str(self.background_path), (bkg / self.bkg_norm).astype(np.uint8)
            )
            logging.info(f"Background saved at {self.background_path}")

    @bkg_model.deleter
    def bkg_model(self):
        self.background_path.unlink(missing_ok=True)

    @property
    def ROI_mask(self) -> np.ndarray | None:
        if self.ROI_mask_path.is_file():
            return cv2.imread(str(self.ROI_mask_path))[..., 0].astype(bool)
        else:
            return None

    @ROI_mask.setter
    def ROI_mask(self, mask: np.ndarray | None):
        if mask is None:
            del self.ROI_mask
        else:
            print("asekjbasdglkbaertgklbndrglkjbsdg")
            cv2.imwrite(str(self.ROI_mask_path), (mask * 255).astype(np.uint8))
            logging.info(f"Background saved at {self.background_path}")

    @ROI_mask.deleter
    def ROI_mask(self):
        self.ROI_mask_path.unlink(missing_ok=True)

    @property
    def multiple_video_paths(self):
        return len(self._video_paths) > 1

    @property
    def use_ROI(self):
        return self.ROI_mask_path.is_file()

    @property
    def episodes(self) -> list[Episode]:
        """
        Indicates the starting and ending frames of each video episode.
        Video episodes are used for parallelization of some processes.
        """
        return self._episodes

    @property
    def video_paths(self) -> list[Path]:
        """List of paths (str) indicate each of the files that compose the
        video

        Returns
        -------
        List[str]
            List of paths to the different files the video is composed of.
            If the video is a single file, the list will have length 1.

        See Also
        --------
        :method:`~idtrackerai.video.Video.get_video_paths`
        """
        return self._video_paths

    @video_paths.setter
    def video_paths(self, video_path):
        self._video_paths = self.process_video_paths(video_path)
        to_print = "Setting video_paths to:"
        for video_path in self._video_paths:
            to_print += f"\n  {video_path}"
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
        return len(self._episodes)

    @property
    def original_width(self):
        """Original video width in pixels.

        Returns
        -------
        int
            Original video width in pixels. It does not consider the resolution
            reduction factor defined by the user.
        """
        return self._original_width

    @property
    def original_height(self):
        """Original video height in pixels.

        Returns
        -------
        int
            Original video width in pixels. It does not consider the resolution
            reduction factor defined by the user.
        """
        return self._original_height

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
        return np.round(self.original_width * self.resolution_reduction).astype(int)

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
        return np.round(self.original_height * self.resolution_reduction).astype(int)

    @property
    def frames_per_second(self):
        """Video frame rate in frames per second.

        Returns
        -------
        int
            Video frame rate in frames per second obtained by OpenCV from the
            video file.
        """
        return self._frames_per_second

    # TODO: move tracker.py
    @property
    def first_frame_first_global_fragment(self):
        return self._first_frame_first_global_fragment

    # TODO: move to crossings_detection.py
    @property
    def median_body_length_full_resolution(self):
        """Median body length in pixels in full frame resolution
        (i.e. without considering the resolution reduction factor)
        """
        return self.median_body_length / self.resolution_reduction

    # Processing steps
    # Flags to indicate whether the different processes have finished or not
    # It was used in the passed for the resume feature, but it is not active
    # in the current version

    @property
    def has_residual_identification(self):
        return self._has_residual_identification

    @property
    def has_impossible_jumps_solved(self):
        return self._has_impossible_jumps_solved

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
    def blobs_path_interpolated(self) -> Path:
        return self.preprocessing_folder / "list_of_blobs_interpolated.npy"

    @property
    def global_fragments_path(self) -> Path:
        """get the path to save the list of global fragments after
        fragmentation"""
        return self.preprocessing_folder / "list_of_global_fragments.pickle"

    @property
    def fragments_path(self) -> Path:
        """get the path to save the list of global fragments after
        fragmentation"""
        return self.preprocessing_folder / "list_of_fragments.pickle"

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

    # Validation
    @property
    def identities_groups(self):
        """Groups of identities stored during the validation of the tracking
        in the validation GUI. This is useful to group identities in different
        classes depending on the experiment.

        This feature was coded becuase some users require indicating classes
        of individuals but we do not use it in the lab.
        """
        return self._identities_groups

    @property
    def is_centroid_updated(self):
        """Indicates whether the (x, y) centroid of some blobs has been updated
        during the validation process in the validation GUI.
        """
        return self._is_centroid_updated

    @is_centroid_updated.setter
    def is_centroid_updated(self, value):
        self._is_centroid_updated = value

    # Methods
    def save(self):
        """Saves the instantiated Video object"""
        logging.info(f"Saving video object in {self.path_to_video_object}")
        self.path_to_video_object.write_text(
            json.dumps(self.__dict__, default=json_default, indent=4)
        )

    @classmethod
    def load(cls, path: Path | str) -> "Video":
        """Load a video object stored in a JSON file"""
        path = Path(path).resolve()
        if not path.is_file():
            path /= "video_object.json"
            if not path.is_file():
                raise FileNotFoundError(path)

        with open(path, "r") as file:
            json_dict = json.load(file, object_hook=json_object_hook)

        video = cls.__new__(cls)
        video.__dict__.update(json_dict)
        video.update_paths(path)
        return video

    def update_paths(self, new_video_object_path: Path):
        """Update paths of objects (e.g. blobs_path, preprocessing_folder...)
        according to the new location of the new video object given
        by `new_video_object_path`.

        Parameters
        ----------
        new_video_object_path : str
            Path to a video_object.npy
        """

        if self.session_folder != new_video_object_path.parent:
            self.session_folder = new_video_object_path.parent
            logging.info(f"Updated session folder to {self.session_folder}")

        try:
            assert_all_files_exist(self.video_paths)
            logging.info(
                f"All video paths found in the original folder {self.video_folder}, "
                "the original video_paths are kept"
            )
        except FileNotFoundError:
            possible_new_video_paths = [
                self.session_folder.parent / path.name for path in self.video_paths
            ]
            assert_all_files_exist(possible_new_video_paths)
            logging.info(
                f"All video paths found in {self.session_folder.parent}, updating Video.video_paths"
            )
            self._video_paths = possible_new_video_paths

        self.save()

    @staticmethod
    def process_video_paths(video_paths):
        accepted_extensions = conf.AVAILABLE_VIDEO_EXTENSION
        assert video_paths, "Empty video_paths list"
        if not isinstance(video_paths, list):
            video_paths = [video_paths]

        return_video_paths = []
        while video_paths:
            path = Path(video_paths.pop()).resolve()
            assert path.exists(), f"Video path {path} not found."
            if path.is_file():
                assert (
                    path.suffix in accepted_extensions
                ), f"Supported video extensions are {accepted_extensions}"
                return_video_paths.append(path)
            elif path.is_dir():
                return_video_paths += sorted(
                    file
                    for file in path.iterdir()
                    if file.suffix in accepted_extensions
                )
            else:
                raise ValueError(
                    f"Video path {path} exists but is either a file nor a dir"
                )

        return return_video_paths

    @staticmethod
    def get_info_from_video_paths(video_paths: list[Path]):
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

        assert len(set(widths)) == 1, "Video paths have different sizes"
        assert len(set(heights)) == 1, "Video paths have different sizes"
        assert len(set(fps)) == 1, "Video paths have different framerates"

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

        self.number_of_accumulated_global_fragments = []
        self.number_of_non_certain_global_fragments = []
        self.number_of_randomly_assigned_global_fragments = []
        self.number_of_nonconsistent_global_fragments = []
        self.number_of_nonunique_global_fragments = []
        self.number_of_acceptable_global_fragments = []
        self.ratio_of_accumulated_images = []

        self.accumulation_statistics_attributes_list = [
            "number_of_accumulated_global_fragments",
            "number_of_non_certain_global_fragments",
            "number_of_randomly_assigned_global_fragments",
            "number_of_nonconsistent_global_fragments",
            "number_of_nonunique_global_fragments",
            "number_of_acceptable_global_fragments",
            "ratio_of_accumulated_images",
        ]

    # TODO: Move to accumulation_manager.py
    def store_accumulation_step_statistics_data(self, new_values):
        [
            getattr(self, attr).append(value)
            for attr, value in zip(
                self.accumulation_statistics_attributes_list, new_values
            )
        ]

    # TODO: Move to accumulation_manager.py
    def store_accumulation_statistics_data(
        self, accumulation_trial, number_of_possible_accumulation=None
    ):
        if number_of_possible_accumulation is None:

            number_of_possible_accumulation = (
                conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS + 1
            )
        if not hasattr(self, "accumulation_statistics"):
            self.accumulation_statistics = [None] * number_of_possible_accumulation
        self.accumulation_statistics[accumulation_trial] = [
            getattr(self, stat_attr)
            for stat_attr in self.accumulation_statistics_attributes_list
        ]

    @staticmethod
    def get_processing_episodes(video_paths, tracking_intervals=None):
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
                if frame_number >= start and frame_number < end:
                    return i
            return None

        # total number of frames for every video path
        video_paths_n_frames = [
            int(cv2.VideoCapture(str(path)).get(7)) for path in video_paths
        ]
        number_of_frames = sum(video_paths_n_frames)

        # set full tracking interval if not defined
        if tracking_intervals is None or tracking_intervals == "all":
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
        limits = sorted(list(set(limits)))

        # Create "long episodes" as the intervals between any video path
        # change or tracking interval change (keeping only the ones that
        # are inside a tracking interval)
        long_episodes = []
        for start, end in zip(limits[:-1], limits[1:]):
            if (
                (in_which_interval(start, tracking_intervals) is not None)
                and start < number_of_frames
                and start >= 0
            ):
                long_episodes.append((start, end))

        # build definitive episodes by dividing long episodes to fit in
        # the conf.FRAMES_PER_EPISODE restriction
        index = 0
        episodes = []
        for start, end in long_episodes:
            video_path_index = in_which_interval(start, video_paths_intervals)
            assert video_path_index is not None
            gloval_local_offset = video_paths_intervals[video_path_index][0]

            n_subepisodes = int((end - start) / (conf.FRAMES_PER_EPISODE + 1))
            new_episode_limits = np.linspace(start, end, n_subepisodes + 2, dtype=int)
            for new_start, new_end in zip(
                new_episode_limits[:-1], new_episode_limits[1:]
            ):
                episodes.append(
                    Episode(
                        index=index,
                        local_start=new_start - gloval_local_offset,
                        local_end=new_end - gloval_local_offset,
                        video_path_index=video_path_index,
                        global_start=new_start,
                        global_end=new_end,
                    )
                )
                index += 1
        return (number_of_frames, video_paths_n_frames, tracking_intervals, episodes)

    @staticmethod
    def in_which_interval(frame_number, intervals):
        for i, (start, end) in enumerate(intervals):
            if frame_number >= start and frame_number < end:
                return i
        return None

    def delete_data(self, data_policy=None):
        """Deletes some folders with data, to make the outcome lighter.

        Which folders are deleted depends on the constant DATA_POLICY
        """
        if data_policy is None:
            data_policy = conf.DATA_POLICY
        logging.info(f"Data policy: {data_policy}")

        if data_policy in [
            "trajectories",
            "validation",
            "knowledge_transfer",
            "idmatcher.ai",
        ]:

            remove_dir(self.segmentation_data_folder)
            remove_file(self.global_fragments_path)
            remove_dir(self.crossings_detector_folder)

        if data_policy in ["trajectories", "validation", "knowledge_transfer"]:
            remove_dir(self.id_images_folder)

        if data_policy in ["trajectories", "validation"]:
            for path in self.session_folder.glob("accumulation_*"):
                remove_dir(path)
            remove_dir(self.session_folder / "pretraining")

        if data_policy == "trajectories":
            remove_dir(self.preprocessing_folder)
