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
from abc import ABC, abstractmethod
import os
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.animals_detection.segmentation import segment
from rich.pretty import pretty_repr
from idtrackerai.utils.py_utils import CheckSegmentationError

# TODO remove ABC, not very useful
class AnimalsDetectionABC(ABC):
    def __init__(self, video: Video):
        """
        This class generates a ListOfBlobs object and updates the video
        object with information about the process.

        Parameters
        ----------
        video: Video
            An instance of the class :class:`~idtrackerai.video.Video`.

        Attributes
        ----------
        video: Video
        list_of_blobs: ListOfBlobs
        detection_parameters: Dict



        See Also
        --------
        :class:`~idtrackerai.list_of_blobs.ListOfBlobs`
        """

        self.video = video
        self.list_of_blobs = None
        self._detection_parameters = None

    def __call__(self):
        self.video._detect_animals_time = time.time()
        self.video.create_preprocessing_folder()
        self.video.create_images_folders()

        # Set detection parameter
        self.set_detection_parameters()

        # Main call
        self.list_of_blobs = self.get_list_of_blobs()
        assert len(self.list_of_blobs) == self.video.number_of_frames

        # Finish animals detection
        self.video._detect_animals_time = (
            time.time() - self.video.detect_animals_time
        )
        self.video._has_animals_detected = True

        self.check_segmentation()
        return self.list_of_blobs

    @property
    def detection_parameters(self):
        return self._detection_parameters

    @abstractmethod
    def set_detection_parameters(self):
        pass

    @abstractmethod
    def get_list_of_blobs(self):
        pass

    @abstractmethod
    def check_segmentation(self):
        pass

    @abstractmethod
    def save_inconsistent_frames(self):
        pass


class AnimalsDetectionAPI(AnimalsDetectionABC):
    # The order of computing mask, bkg_model and resolution_reduction
    # is important:
    # 1. The mask affects the computation of the frame average intensity
    # that is used during the computation of the background model
    # 2. When setting the resolution_reduction, the mask and the bkg_model
    # are resized accordingly
    detection_parameters_keys = [
        "intensity_ths",
        "area_ths",
        # "apply_ROI",
        # "rois",
        "ROI_mask",
        "use_bkg",
        "bkg_model",
        "resolution_reduction",
        "tracking_intervals",
    ]

    def __init__(self, video: Video):
        super().__init__(video)
        # These attributes are stored in each blob for other purposes
        # TODO: ideally each blob does not need to store these values
        self._attributes_to_store_in_each_blob = {
            "width": self.video.width,
            "height": self.video.height,
            "number_of_animals": self.video.number_of_animals,
        }

    def set_detection_parameters(self):
        self._detection_parameters = {
            key: getattr(self.video, key)
            for key in self.detection_parameters_keys
        }

        logging.info(
            f"Detection parameters are:\n"
            + pretty_repr(self.detection_parameters)
        )

    @property
    def attributes_to_store_in_each_blob(self):
        return self._attributes_to_store_in_each_blob

    def get_list_of_blobs(self):
        """
        Segments the video returning a ListOfBlobs object

        Returns
        -------
        list_of_blobs: ListOfBlobs

        See Also
        --------
        :class:`~idtrackerai.list_of_blobs.ListOfBlobs`
        """

        logging.info("Segmenting video")
        # TODO pass explicitly detection_parameters
        blobs_in_video = segment(
            self.detection_parameters,
            self.attributes_to_store_in_each_blob,
            self.video.episodes,
            self.video.segmentation_data_foler,
            self.video.video_paths,
            self.video.number_of_frames,
        )

        logging.info("Generating ListOfBlobs object")

        return ListOfBlobs(blobs_in_video=blobs_in_video)

    def check_segmentation(self):
        """
        idtracker.ai is designed to work under the assumption that all the
        detected blobs are animals. In the frames where the number of
        detected blobs is higher than the number of animals in the video, it is
        likely that some blobs do not represent animals. In this scenario
        idtracker.ai might misbehave. This method allows to check such
        condition.

        Returns
        -------
        consistent_segmentation: bool
            True if the number of blobs detected in each frame of the video
            is smaller than the number of animals in the video as specified by
            the user. Otherwise it returns False.
        """
        logging.info("Checking segmentation")

        maximum_number_of_blobs = 0
        frames_with_more_blobs_than_animals = []
        for frame_number, blobs_in_frame in enumerate(
            self.list_of_blobs.blobs_in_video
        ):
            n_blobs = len(blobs_in_frame)
            if n_blobs > self.video.number_of_animals:
                frames_with_more_blobs_than_animals.append(frame_number)
            maximum_number_of_blobs = max(n_blobs, maximum_number_of_blobs)

        self.video._frames_with_more_blobs_than_animals = (
            frames_with_more_blobs_than_animals
        )
        self.video._maximum_number_of_blobs = maximum_number_of_blobs

        n_error_frames = len(frames_with_more_blobs_than_animals)
        logging.info(
            f"There are {n_error_frames} frames with more blobs than animals"
        )
        if n_error_frames > 0:
            logging.warning(
                "This can be detrimental for the proper functioning of the system."
            )
            if n_error_frames < 25:
                logging.warning(
                    f"Frames with more blobs than animals: {frames_with_more_blobs_than_animals}"
                )
            else:
                logging.warning(
                    "Too much frames with more blobs than animals"
                    "for printing their indexes in log"
                )

            self.save_inconsistent_frames()
            if self.video.check_segmentation:
                self.list_of_blobs.save(self.video.blobs_path)
                raise CheckSegmentationError(
                    f"check_segmentation is {True}, exiting...\n"
                    "Please readjust the segmentation parameters and track again"
                )
            else:
                logging.info(
                    f"check_segmentation is {False}, ignoring the above errors"
                )

    def save_inconsistent_frames(self):
        """
        Saves a .csv file with the frame number of the frames that had
        more segmented blobs than animals.

        Returns
        -------
        outfile_path: str
            The path to the .csv file
        """
        outfile_path = os.path.join(
            self.video.session_folder, "inconsistent_frames.csv"
        )
        logging.info(
            f"Saving indexes of frames with more blobs than animals as {outfile_path}"
        )
        with open(outfile_path, "w") as outfile:
            outfile.write(
                "\n".join(
                    map(
                        str,
                        self.video.frames_with_more_blobs_than_animals,
                    )
                )
            )
