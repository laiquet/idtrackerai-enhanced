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

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted

from idtrackerai.animals_detection.segmentation import compute_background
from idtrackerai.tracker.tracker import TrackerAPI
from idtrackerai.utils import Episode, conf
from idtrackerai.utils.py_utils import (
    assert_all_files_exist,
    build_ROI_mask_from_list,
    create_dir,
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

    # TODO remove these defaults, they are already in __main__
    def __init__(
        self,
        video_paths,
        number_of_animals,
        intensity_ths,
        area_ths,
        ROI_list=None,
        session="no_name",
        tracking_intervals=None,
        resolution_reduction=1,
        ROI_mask=None,
        use_bkg=False,
        bkg_model=None,
        setup_points=None,
        track_wo_identities=False,
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

        logging.debug("Video object init")
        self.check_segmentation = check_segmentation
        self.setup_points = setup_points
        self.track_wo_identities = track_wo_identities
        self.intensity_ths = intensity_ths
        self.area_ths = area_ths
        self.knowledge_transfer_folder = knowledge_transfer_folder
        self.resolution_reduction = resolution_reduction
        self.number_of_animals = int(number_of_animals)
        self.video_paths = video_paths  # has a setter
        self.tracking_intervals = tracking_intervals
        self.sigma_gaussian_blurring = sigma_gaussian_blurring

        if self.knowledge_transfer_folder:
            self.knowledge_transfer_folder = Path(
                self.knowledge_transfer_folder
            )
            assert (
                self.knowledge_transfer_folder.exists()
            ), f"{self.knowledge_transfer_folder} not found"

        (
            self._original_width,
            self._original_height,
            self._frames_per_second,
        ) = self.get_info_from_video_paths(self.video_paths)
        (
            self._number_of_frames,
            _,
            self.tracking_intervals,
            self._episodes,
        ) = self.get_processing_episodes(
            self.video_paths, self.tracking_intervals
        )

        logging.info(f"The video has {self.number_of_frames} frames")
        logging.info(f"The video has {self.number_of_episodes} episodes:")
        for e in self.episodes:
            video_name = self.video_paths[e.video_path_index].name
            logging.info(
                f"\tEpisode {e.index}, frames ({e.local_start} => {e.local_end}) of /{video_name}"
            )
        assert self.number_of_episodes > 0

        if ROI_list:
            self.original_ROI = build_ROI_mask_from_list(
                self.original_width,
                self.original_height,
                list_of_ROIs=ROI_list,
            )
        else:
            self.original_ROI = ROI_mask

        self.ROI_mask = self.original_ROI

        if use_bkg:
            if bkg_model is not None:
                logging.info("Storing previously computed background model")
                self.bkg_model = bkg_model
            else:
                self.bkg_model = compute_background(
                    self.video_paths,
                    self.original_ROI,
                    self.episodes,
                )
        else:
            logging.info("No background model computed")
            self.bkg_model = None

        if identity_transfer:
            # TODO: the id_image_size is not really passed by
            # the used but inferred from the knowledge transfer folder
            (
                self.identity_transfer,
                self._id_image_size,
            ) = TrackerAPI.check_if_identity_transfer_is_possible(
                self.number_of_animals,
                self.knowledge_transfer_folder,
            )
        else:
            self.identity_transfer = False
            self._id_image_size = None

        if conf.output_folder:
            self._session_folder = (
                conf.output_folder / f"session_{session.strip()}"
            )
        else:
            self._session_folder = (
                self.video_folder / f"session_{session.strip()}"
            )
        self.create_session_folder()

        # TODO: HARDCODED _number_of_channels. Change if color information is used.
        # Currently idtracker.ai does not rely on color. All color videos
        # are converted to gray scale so the number of channels is forced to
        # always be one. This should be changed if color images are used for
        # identification, as this attributed is used to created the
        # identification images.
        self._number_of_channels = 1  # Used to create identification images

        # Attributes computed by other processes in the tracking
        # During crossing detection
        self._median_body_length = None  # updated later
        self._model_area = None  # updated later
        self._there_are_crossings = True  # updated later
        # During fragmentation
        self._fragment_identifier_to_index = None  # updated later
        # During tracking (protocol cascade)
        self._identity_transfer = None  # updated later
        self._tracking_with_knowledge_transfer = False  # updated later
        self._percentage_of_accumulated_images = None  # updated later
        self._first_frame_first_global_fragment = []  # updated later
        self._accumulation_trial = 0  # updated later
        self._knowledge_transfer_info_dict = None  # updated later
        # During validation (in validation GUI)
        self._identities_groups = {}  # updated later
        # self.accumulation_iteration = 0
        self._accumulation_folder = None

        # Flag to decide which type of interpolation is done. This flag
        # is updated when we update a blob centroid
        self._is_centroid_updated = False
        self._estimated_accuracy = None

        # Processes states
        self._has_preprocessing_parameters = False
        self._has_animals_detected = False  # animal detection and segmentation
        self._has_crossings_detected = False  # crossings detection
        self._has_been_fragmented = False  # fragmentation
        self._has_protocol1_finished = False  # protocols cascade
        self._has_protocol2_finished = False  # protocols cascade
        self._has_protocol3_pretraining_finished = False  # protocols cascade
        self._has_protocol3_accumulation_finished = False  # protocols cascade
        self._has_protocol3_finished = False  # protocols cascade
        self._has_residual_identification = False  # residual identification
        self._has_impossible_jumps_solved = False  # post-processing
        self._has_crossings_solved = False  # crossings interpolation
        self._has_trajectories = False  # trajectories generation
        self._has_trajectories_wo_gaps = False  # trajectories generation

        # Timers
        self._detect_animals_time = 0.0
        self._crossing_detector_time = 0.0
        self._fragmentation_time = 0.0
        self._protocol1_time = 0.0
        self._protocol2_time = 0.0
        self._protocol3_pretraining_time = 0.0
        self._protocol3_accumulation_time = 0.0
        self._identify_time = 0.0
        self._create_trajectories_time = 0.0

    @property
    def multiple_video_paths(self):
        return len(self._video_paths) > 1

    @property
    def use_ROI(self):
        return self.original_ROI is not None

    @property
    def use_bkg(self):
        return self.bkg_model is not None

    # General video properties
    @property
    def number_of_channels(self):
        """Number of channels in the video"""
        return self._number_of_channels

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
    def number_of_frames(self):
        """Total number of frames in the video to be tracked.

        Returns
        -------
        int
            Total number of frames in the video to be tracked. It considers
            all frames in all episodes. If the video consists of different
            files, the sum of the number of frames of all files is considered.

        See Also
        --------
        :method:`~idtrackerai.video.Video.get_num_frames_and_processing_episodes`
        """
        return self._number_of_frames

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
        return np.round(
            self.original_width * self.resolution_reduction
        ).astype(int)

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
        return np.round(
            self.original_height * self.resolution_reduction
        ).astype(int)

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

    # TODO: Not used. Check if necessary. Otherwise delete.
    @property
    def fragment_identifier_to_index(self):
        return self._fragment_identifier_to_index

    # TODO: move to accumulation_manager.py
    @property
    def percentage_of_accumulated_images(self):
        return self._percentage_of_accumulated_images

    # TODO: move to constants.py
    @property
    def erosion_kernel_size(self):
        return self._erosion_kernel_size

    # TODO: move to accumulation_manager.py
    @property
    def accumulation_trial(self):
        return self._accumulation_trial

    # TODO: move to tracker.py
    @property
    def estimated_accuracy(self):
        return self._estimated_accuracy

    # TODO: move to crossings_detection.py
    @property
    def id_image_size(self):
        return self._id_image_size

    # TODO: Probably not used. Check and delete
    @property
    def knowledge_transfer_info_dict(self):
        return self._knowledge_transfer_info_dict

    # TODO: move tracker.py
    @property
    def first_frame_first_global_fragment(self):
        return self._first_frame_first_global_fragment

    # TODO: move to crossings_detection.py where it is computed
    @property
    def median_body_length(self):
        """Median body length in pixels considering the resolution reduction
        factor
        """
        return self._median_body_length

    # TODO: move to crossings_detection.py
    @property
    def median_body_length_full_resolution(self):
        """Median body length in pixels in full frame resolution
        (i.e. without considering the resolution reduction factor)
        """
        return self.median_body_length / self.resolution_reduction

    # TODO: move to crossings_detection.py
    @property
    def model_area(self):
        return self._model_area

    # TODO: move to crossings_detection.py
    @property
    def there_are_crossings(self):
        return self._there_are_crossings

    # TODO: move to accumulation_manager.py
    @property
    def ratio_accumulated_images(self):
        return self._ratio_accumulated_images

    # Processing steps
    # Flags to indicate whether the different processes have finished or not
    # It was used in the passed for the resume feature, but it is not active
    # in the current version
    @property
    def has_animals_detected(self):
        return self._has_animals_detected

    @property
    def has_crossings_detected(self):
        return self._has_crossings_detected

    @property
    def has_been_fragmented(self):
        return self._has_been_fragmented

    @property
    def has_protocol1_finished(self):
        return self._has_protocol1_finished

    @property
    def has_protocol2_finished(self):
        return self._has_protocol2_finished

    @property
    def has_protocol3_pretraining_finished(self):
        return self._has_protocol3_pretraining_finished

    @property
    def has_protocol3_accumulation_finished(self):
        return self._has_protocol3_accumulation_finished

    @property
    def has_protocol3_finished(self):
        return self._has_protocol3_finished

    @property
    def has_residual_identification(self):
        return self._has_residual_identification

    @property
    def has_impossible_jumps_solved(self):
        return self._has_impossible_jumps_solved

    @property
    def has_crossings_solved(self):
        return self._has_crossings_solved

    @property
    def has_trajectories(self):
        return self._has_trajectories

    @property
    def has_trajectories_wo_gaps(self):
        return self._has_trajectories_wo_gaps

    @property
    def has_preprocessing_parameters(self):
        return self._has_preprocessing_parameters

    # Attributes to store computational times of the different processses
    # TODO: each process class should have its own attribute to store this.
    @property
    def detect_animals_time(self):
        return self._detect_animals_time

    @property
    def crossing_detector_time(self):
        return self._crossing_detector_time

    @property
    def fragmentation_time(self):
        return self._fragmentation_time

    @property
    def protocol1_time(self):
        return self._protocol1_time

    @property
    def protocol2_time(self):
        return self._protocol2_time

    @property
    def protocol3_pretraining_time(self):
        return self._protocol3_pretraining_time

    @property
    def protocol3_accumulation_time(self):
        return self._protocol3_accumulation_time

    @property
    def identify_time(self):
        return self._identify_time

    @property
    def create_trajectories_time(self):
        return self._create_trajectories_time

    # Paths and folders
    # TODO: The different processes should create and store the path to the
    # folder where they save the data
    @property
    def preprocessing_folder(self) -> Path:
        return self.session_folder / "preprocessing"

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
    def accumulation_folder(self) -> Path:
        return self._accumulation_folder
        # FIXME
        # return self.session_folder / f"accumulation_{self.accumulation_trial}"

    @property
    def id_images_folder(self) -> Path:
        return self.session_folder / "identification_images"

    # TODO: This should probably be the only path that should be stored in
    # Video.
    @property
    def session_folder(self):
        return self._session_folder

    @property
    def blobs_path(self) -> Path:
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been succesfully performed"""
        return self.preprocessing_folder / "blobs_collection.npy"

    @property
    def blobs_path_segmented(self) -> Path:
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been succesfully performed"""
        return self.preprocessing_folder / "blobs_collection_segmented.npy"

    @property
    def blobs_path_interpolated(self) -> Path:
        return self.preprocessing_folder / "blobs_collection_interpolated.npy"

    @property
    def trajectories_wo_identification_folder(self) -> Path:
        return self.session_folder / "trajectories_wo_identification"

    @property
    def trajectories_wo_gaps_folder(self) -> Path:
        return self.session_folder / "trajectories_wo_gaps"

    @property
    def global_fragments_path(self) -> Path:
        """get the path to save the list of global fragments after
        fragmentation"""
        return self.preprocessing_folder / "global_fragments.npy"

    @property
    def fragments_path(self) -> Path:
        """get the path to save the list of global fragments after
        fragmentation"""
        return self.preprocessing_folder / "fragments.npy"

    @property
    def path_to_video_object(self) -> Path:
        return self.session_folder / "video_object.npy"

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
        """Saves the instantiated Video object.

        This is not good practices, as we are saving an object. We should be
        saving a dictionary and reconstruct the object from it in the load
        method.
        """
        # TODO: Do not save full objects. Save ad dictionary and reconstruct
        # the object in the load method.
        logging.info(f"Saving video object in {self.path_to_video_object}")
        np.save(self.path_to_video_object, self)

    @staticmethod
    def load(video_object_path: Path | str) -> Video:
        """Load a video object stored in a .npy file.

        In the future it should load a json file with information about the
        video and reconstruct the Video object from it.
        """
        video_object_path = Path(video_object_path).resolve()
        if not video_object_path.is_file():
            video_object_path /= "video_object.npy"
            if not video_object_path.is_file():
                raise FileNotFoundError(video_object_path)

        video_object = np.load(video_object_path, allow_pickle=True).item()
        video_object.update_paths(video_object_path)
        return video_object

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
            self._session_folder = new_video_object_path.parent
            logging.info(f"Updated session folder to {self.session_folder}")

        try:
            assert_all_files_exist(self.video_paths)
            logging.info(
                f"All video paths found in the original folder {self.video_folder}."
                "We will keep the original video_path"
            )
        except FileNotFoundError:
            possible_new_video_paths = [
                self.session_folder.parent / path.name
                for path in self.video_paths
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
            path = Path(video_paths.pop())
            assert path.exists(), f"Video path {path} not found."
            if path.is_file():
                assert (
                    path.suffix in accepted_extensions
                ), f"Supported video extensions are {accepted_extensions}"
                return_video_paths.append(path)
            elif path.is_dir():
                return_video_paths += natsorted(
                    [
                        file
                        for file in path.iterdir()
                        if file.suffix in accepted_extensions
                    ]
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

    # TODO: move to crossings_detection.py
    def compute_id_image_size(self, maximum_body_length):
        """Uses an estimate of the body length of the animals in order to
        compute the size of the square image that is generated from every
        blob to identify the animals
        """
        if self.id_image_size is None:
            id_image_size = int(maximum_body_length / np.sqrt(2))
            id_image_size += id_image_size % 2
            self._id_image_size = (
                id_image_size,
                id_image_size,
                self.number_of_channels,
            )

    # Methods to create folders where to store data
    # TODO: Some of these methods should go to the classes corresponding to
    # the process.
    def create_session_folder(self):
        """Creates a folder where all the results of the tracking session
        will be stored.
        """
        create_dir(self.session_folder)

    # TODO: It should be fragmented and moved to animals_detection.py and
    # crossings_detection.py. One for segmentation_data and other to
    # identification_images.
    def create_images_folders(self):
        """Creates folders to store segmentation images and identification
        images.
        """
        create_dir(self.segmentation_data_folder)
        create_dir(self.id_images_folder)

    def create_preprocessing_folder(self):
        """If it does not exist creates a folder called preprocessing
        in the video folder"""
        create_dir(self.preprocessing_folder)

    def create_crossings_detector_folder(self):
        """If it does not exist creates a folder called crossing_detector
        in the video folder"""
        create_dir(self.crossings_detector_folder)

    def create_pretraining_folder(self, delete=False):
        """Creates a folder named pretraining in video_folder where the model
        trained during the pretraining is stored
        """
        create_dir(self.pretraining_folder, remove_existing=delete)

    def create_accumulation_folder(self, iteration_number=None, delete=False):
        """Folder in which the model generated while accumulating is stored
        (after pretraining)
        """
        if iteration_number is None:
            iteration_number = self.accumulation_trial
        self._accumulation_folder = (
            self.session_folder / f"accumulation_{iteration_number}"
        )
        # FIXME
        create_dir(self.accumulation_folder, remove_existing=delete)

    def create_individual_videos_folder(self):
        """Create folder where to save the individual videos"""
        create_dir(self.individual_videos_folder)

    def create_trajectories_folder(self):
        """Folder in which trajectories files are stored"""
        create_dir(self.trajectories_folder)

    def create_trajectories_wo_identification_folder(self):
        """Folder in which trajectories without identites are stored"""
        create_dir(self.trajectories_wo_identification_folder)

    def create_trajectories_wo_gaps_folder(self):
        """Folder in which trajectories files are stored"""
        create_dir(self.trajectories_wo_gaps_folder)

    # Some methods related to the accumulation process
    # TODO: Move to accumulation_manager.py
    def init_accumulation_statistics_attributes(self, attributes=None):
        if attributes is None:
            attributes = [
                "number_of_accumulated_global_fragments",
                "number_of_non_certain_global_fragments",
                "number_of_randomly_assigned_global_fragments",
                "number_of_nonconsistent_global_fragments",
                "number_of_nonunique_global_fragments",
                "number_of_acceptable_global_fragments",
                "ratio_of_accumulated_images",
            ]
        self.accumulation_statistics_attributes_list = attributes
        [
            setattr(self, attribute, [])
            for attribute in self.accumulation_statistics_attributes_list
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
        self,
        accumulation_trial,
        number_of_possible_accumulation=None,
    ):
        if number_of_possible_accumulation is None:

            number_of_possible_accumulation = (
                conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS + 1
            )
        if not hasattr(self, "accumulation_statistics"):
            self.accumulation_statistics = [
                None
            ] * number_of_possible_accumulation
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
        tracking_intervals_changes = list(
            np.asarray(tracking_intervals).flatten()
        )

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
            gloval_local_offset = video_paths_intervals[video_path_index][0]

            n_subepisodes = int((end - start) / (conf.FRAMES_PER_EPISODE + 1))
            new_episode_limits = np.linspace(
                start, end, n_subepisodes + 2, dtype=int
            )
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
        return (
            number_of_frames,
            video_paths_n_frames,
            tracking_intervals,
            episodes,
        )

    @staticmethod
    def in_which_interval(frame_number, intervals):
        for i, (start, end) in enumerate(intervals):
            if frame_number >= start and frame_number < end:
                return i
        return None

    # def in_which_episode(self, frame_number: int):
    #     """Given a `frame_number` of the whole video it returns the episode
    #     number.

    #     Parameters
    #     ----------
    #     frame_number : int
    #         Frame number considering all frames of the video.

    #     Returns
    #     -------
    #     int
    #         Episode number where the `frame_number` corresponds to.
    #     """
    #     for episode in self.episodes:
    #         if (
    #             frame_number >= episode.global_start
    #             and frame_number < episode.global_end
    #         ):
    #             return episode.index
    #     return None

    # TODO: move to tracker.py
    def compute_estimated_accuracy(self, fragments):
        weighted_P2 = 0
        number_of_individual_blobs = 0

        for fragment in fragments:
            if fragment.is_an_individual:
                if fragment.assigned_identities[0] != 0:
                    weighted_P2 += (
                        fragment.P2_vector[fragment.assigned_identities[0] - 1]
                        * fragment.number_of_images
                    )
                number_of_individual_blobs += fragment.number_of_images

        self._estimated_accuracy = weighted_P2 / number_of_individual_blobs

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
            remove_file(self.blobs_path_segmented)
            remove_dir(self.crossings_detector_folder)

        if data_policy in [
            "trajectories",
            "validation",
            "knowledge_transfer",
        ]:
            remove_dir(self.id_images_folder)

        if data_policy in ["trajectories", "validation"]:
            for path in self.session_folder.glob("accumulation_*"):
                remove_dir(path)
            remove_dir(self.session_folder / "pretraining")

        if data_policy == "trajectories":
            remove_dir(self.preprocessing_folder)

    # TODO: to list_of_global_fragments.py, list_of_blobs.py, or tracker.py
    def get_first_frame(self, list_of_blobs):
        if self.number_of_animals != 1:
            return self.first_frame_first_global_fragment[
                self.accumulation_trial
            ]
        elif self.number_of_animals == 1:
            return 0
        else:
            for blobs_in_frame in list_of_blobs.blobs_in_video:
                if len(blobs_in_frame) != 0:
                    return blobs_in_frame[0].frame_number
