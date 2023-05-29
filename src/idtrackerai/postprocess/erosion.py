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

import cv2
import numpy as np

from idtrackerai import Blob, Video


def compute_erosion_disk(blobs_in_video: list[list[Blob]]) -> int:
    min_frame_distance_transform = []
    for blobs_in_frame in blobs_in_video:
        if blobs_in_frame:
            min_frame_distance_transform.append(
                compute_min_frame_distance_transform(blobs_in_frame)
            )

    return np.ceil(np.nanmedian(min_frame_distance_transform)).astype(int)
    # return np.ceil(np.nanmedian([compute_min_frame_distance_transform(video, blobs_in_frame)
    #                              for blobs_in_frame in blobs_in_video
    #                              if len(blobs_in_frame) > 0])).astype(np.int)


def compute_min_frame_distance_transform(blobs_in_frame: list[Blob]) -> float:
    max_distance_transform = []
    for blob in blobs_in_frame:
        if blob.is_an_individual:
            try:
                max_distance_transform.append(
                    np.max(
                        cv2.distanceTransform(
                            blob.get_bbox_mask(), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
                        )
                    )
                )
            except cv2.error:
                logging.warning("Could not compute distance transform for this blob")
    return np.min(max_distance_transform) if max_distance_transform else np.nan


def get_eroded_blobs(
    video: Video, blobs_in_frame: list[Blob], frame_number: int
) -> list[Blob]:
    segmented_frame = np.zeros((video.height, video.width), np.uint8)

    for blob in blobs_in_frame:
        segmented_frame = cv2.fillPoly(segmented_frame, (blob.contour,), 255)

    segmented_eroded_frame = cv2.erode(
        src=segmented_frame,
        kernel=np.ones(video.erosion_kernel_size, np.uint8),
        iterations=1,
    )

    # Extract blobs info
    contours = cv2.findContours(
        segmented_eroded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )[0]

    return [
        Blob(contour, frame_number=frame_number, pixels_are_from_eroded_blob=True)
        for contour in contours
    ]
