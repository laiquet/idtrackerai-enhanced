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
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from idtrackerai import Video

from idtrackerai import ListOfBlobs
from idtrackerai.animals_detection.segmentation import segment
from idtrackerai.utils.py_utils import CheckSegmentationError


class AnimalsDetectionAPI:
    # The order of computing mask, bkg_model and resolution_reduction
    # is important:
    # 1. The mask affects the computation of the frame average intensity
    # that is used during the computation of the background model
    # 2. When setting the resolution_reduction, the mask and the bkg_model
    # are resized accordingly
    detection_parameters_keys = [
        "intensity_ths",
        "area_ths",
        "ROI_mask",
        "use_bkg",
        "bkg_model",
        "resolution_reduction",
    ]

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
        start = time.perf_counter()
        self.video.create_preprocessing_folder()
        self.video.create_images_folders()

        # Set detection parameter
        self.detection_parameters = {
            key: getattr(self.video, key)
            for key in self.detection_parameters_keys
        }

        # Main call
        blobs_in_video = segment(
            self.detection_parameters,
            self.video.episodes,
            self.video.segmentation_data_folder,
            self.video.video_paths,
            self.video.number_of_frames,
        )

        self.list_of_blobs = ListOfBlobs(blobs_in_video)

        assert len(self.list_of_blobs) == self.video.number_of_frames

        # Finish animals detection
        self.video._detect_animals_time = time.perf_counter() - start
        self.video._has_animals_detected = True

        self.check_segmentation()
        self.list_of_blobs.save(self.video.blobs_path)
        return self.list_of_blobs

    def check_segmentation(self):
        """
        idtracker.ai is designed to work under the assumption that all the
        detected blobs are animals. In the frames where the number of
        detected blobs is higher than the number of animals in the video, it is
        likely that some blobs do not represent animals. In this scenario
        idtracker.ai might misbehave. This method allows to check such
        condition.
        """
        logging.info("Checking segmentation")

        error_frames = [
            frame
            for frame, blobs in enumerate(self.list_of_blobs.blobs_in_video)
            if len(blobs) > self.video.number_of_animals
        ]

        self.frames_with_more_blobs_than_animals = error_frames

        n_error_frames = len(error_frames)
        logging.log(
            logging.WARNING if n_error_frames else logging.INFO,
            f"There are {n_error_frames} frames with more blobs than animals",
        )

        if n_error_frames:
            logging.warning(
                "This can be detrimental for the proper functioning of the system"
            )
            if n_error_frames < 25:
                logging.warning(
                    f"Frames with more blobs than animals: {error_frames}"
                )
            else:
                logging.warning(
                    "Too much frames with more blobs than animals "
                    "for printing their indexes in log"
                )

            output_path = self.video.session_folder / "inconsistent_frames.csv"
            logging.info(
                f"Saving indexes of frames with more blobs than animals in {output_path}"
            )
            output_path.write_text("\n".join(map(str, error_frames)))

            if self.video.check_segmentation:
                self.list_of_blobs.save(self.video.blobs_path)
                raise CheckSegmentationError(
                    f"Check_segmentation is {True}, exiting...\n"
                    "Please readjust the segmentation parameters and track again"
                )
            else:
                logging.info(
                    f"Check_segmentation is {False}, ignoring the above errors"
                )
