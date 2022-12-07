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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from idtrackerai import ListOfBlobs, Video

import logging
from math import sqrt

from idtrackerai.crossings_detection.crossing_detector import detect_crossings
from idtrackerai.crossings_detection.model_area import (
    ModelArea,
    compute_body_length,
)


class CrossingsDetectionAPI:
    """
    This crossings detector works under the following assumptions
        1. The number of animals in the video is known (given by the user)
        2. There are frames in the video where all animals are separated from
        each other.
        3. All animals have a similar size
        4. The frame rate of the video is higher enough so that consecutive
        segmented blobs of pixels of the same animal overlap, i.e. some of the
        pixels representing the animal A in frame i are the same in the
        frame i+1.

    NOTE: This crossing detector sets the identification images that will be
    used to identify the animals
    """

    def __init__(self, video: Video, list_of_blobs: ListOfBlobs):
        """
        Classifies all the blobs in list_of_blobs as individuals or crossings
        """
        self.video = video
        self.list_of_blobs = list_of_blobs

    def __call__(self):
        self.video.crossing_detector_time.tic()
        self.video.create_crossings_detector_folder()
        self._estimate_single_indiviual_size()
        self.set_id_images()
        self.list_of_blobs.compute_overlapping_between_subsequent_frames()
        self._train_and_apply_crossing_detector()
        assert len(self.list_of_blobs) == self.video.number_of_frames
        self.video.crossing_detector_time.tac()

    def _estimate_single_indiviual_size(self):
        """
        Computes a model_area of the size of single animals using frames of the
        video where all animals are separated from each other. In these frames
        the number of segmented blobs is the same as the number of animals in
        the video. So, all blobs are individual animals.

        It also estimates them median_body_length of single individuals.

        See Also
        --------
        :class:`~idtrackerai.crossigns_detection.model_area.ModelArea`
        """

        self.model_area = ModelArea(
            self.list_of_blobs.blobs_in_video,
            self.video.number_of_animals,
        )
        self.video.model_area = self.model_area

        self.median_body_length = compute_body_length(
            self.list_of_blobs.blobs_in_video,
            self.video.number_of_animals,
        )
        self.video.median_body_length = self.median_body_length

    def set_id_images(self):
        """
        Creates an square image that we call "identification_image". This
        image is used both to classify the blob as crossing or individual
        and to identify the animals later on in the tracking.
        The length of the diagonal of the identification_image equals the
        medial_body_length
        """
        logging.info("Creating identification images")

        if not self.video.id_image_size:
            id_image_size = int(self.median_body_length / sqrt(2))
            id_image_size += id_image_size % 2
            self.video._id_image_size = [
                id_image_size,
                id_image_size,
                self.video.number_of_channels,
            ]
        else:
            logging.info(
                "Getting identification image size from previous session"
            )
        logging.info(
            f"Identification image size set to {id_image_size}x{id_image_size}"
        )
        self.list_of_blobs.set_images_for_identification(
            self.video.episodes,
            self.video.id_images_file_paths,
            self.video.id_image_size,
        )

    def _train_and_apply_crossing_detector(self):
        """
        Detects all blobs in the video as crossings or individuals
        """
        if self.video.number_of_animals > 1:
            detect_crossings(
                self.list_of_blobs,
                self.video,
                self.model_area,
            )
        else:
            self.video.there_are_crossings = False
