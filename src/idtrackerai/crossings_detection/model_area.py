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
import logging

import numpy as np

from idtrackerai import ListOfBlobs
from idtrackerai.utils import CustomError, conf


class ModelArea:
    """Model of the area used to perform a first discrimination between blobs
    representing single individual and multiple touching animals (crossings)

    Attributes
    ----------

    median : float
        median of the area of the blobs segmented from portions of the video in
        which all the animals are visible (not touching)
    mean : float
        mean of the area of the blobs segmented from portions of the video in
        which all the animals are visible (not touching)
    std : float
        standard deviation of the area of the blobs segmented from portions of
        the video in which all the animals are visible (not touching)
    std_tolerance : int
        tolerance factor

    Methods
    -------
    __call__:
      some description
    """

    def __init__(self, list_of_blobs: ListOfBlobs, number_of_animals: int):
        """computes the median and standard deviation of the area of all the blobs
        in the the video and the median of the the diagonal of the bounding box.
        """
        # areas are collected throughout the entire video inthe cores of the
        # global fragments
        logging.info(
            "Initializing ModelArea for individual/crossing blob initial classification"
        )
        if number_of_animals > 0:
            areas = []
            for blobs_in_frame in list_of_blobs.blobs_in_video:
                if len(blobs_in_frame) == number_of_animals:
                    for blob in blobs_in_frame:
                        areas.append(blob.area)
        else:
            areas = [b.area for b in list_of_blobs.all_blobs]
        areas = np.asarray(areas)

        n_blobs = len(areas)
        if n_blobs == 0:
            raise CustomError(
                "There is not part in the video where the "
                f"{number_of_animals} animals are visible. "
                "Try a different segmentation or check the "
                "number of animals in the video."
            )
        self.median = np.median(areas)
        self.mean = np.mean(areas)
        self.std = np.std(areas)
        self.std_tolerance = conf.MODEL_AREA_SD_TOLERANCE
        self.tolerance = self.std_tolerance * self.std
        logging.info(
            f"Model area computed with {n_blobs} blobs. "
            f"Mean area = {self.mean:.1f}, median = {self.median:.1f}, "
            f"and std = {self.std:.1f} (in pixels)"
        )

    def __call__(self, area) -> bool:
        return (area - self.median) < self.tolerance


def compute_body_length(list_of_blobs: ListOfBlobs, number_of_animals: int) -> float:
    """computes the median of the the diagonal of the bounding box."""
    # areas are collected throughout the entire video in the cores of
    # the global fragments
    if number_of_animals > 0:
        body_lengths = []
        for blobs_in_frame in list_of_blobs.blobs_in_video:
            if len(blobs_in_frame) == number_of_animals:
                for blob in blobs_in_frame:
                    body_lengths.append(blob.estimated_body_length)
    else:
        body_lengths = [b.estimated_body_length for b in list_of_blobs.all_blobs]

    median = np.median(body_lengths)
    logging.info(f"Median body length: {median} pixels")
    return float(median)
    # return np.percentile(body_lengths, 80)
