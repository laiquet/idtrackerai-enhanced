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
from idtrackerai.utils import conf

from idtrackerai import Blob

""" erosion """


def compute_erosion_disk(blobs_in_video: list[list[Blob]]):
    min_frame_distance_transform = []
    for blobs_in_frame in blobs_in_video:
        if len(blobs_in_frame) > 0:
            min_frame_distance_transform.append(
                compute_min_frame_distance_transform(blobs_in_frame)
            )

    return np.ceil(np.nanmedian(min_frame_distance_transform)).astype(int)
    # return np.ceil(np.nanmedian([compute_min_frame_distance_transform(video, blobs_in_frame)
    #                              for blobs_in_frame in blobs_in_video
    #                              if len(blobs_in_frame) > 0])).astype(np.int)


def compute_min_frame_distance_transform(blobs_in_frame: list[Blob]):
    max_distance_transform = []
    for blob in blobs_in_frame:
        if blob.is_an_individual:
            try:
                max_distance_transform.append(
                    compute_max_distance_transform(blob)
                )
            except cv2.error:
                logging.warning(
                    "Could not compute distance transform for this blob"
                )

    # max_distance_transform = [compute_max_distance_transform(video, blob)
    #                           for blob in blobs_in_frame
    #                           if blob.is_an_individual]
    return (
        np.min(max_distance_transform)
        if len(max_distance_transform) > 0
        else np.nan
    )


def generate_temp_image(contour, bounding_box_in_frame_coordinates):
    temp_image = np.zeros(
        (
            bounding_box_in_frame_coordinates[1][1]
            - bounding_box_in_frame_coordinates[0][1],
            bounding_box_in_frame_coordinates[1][0]
            - bounding_box_in_frame_coordinates[0][0],
        ),
        np.uint8,
    )

    temp_image = cv2.fillPoly(
        img=temp_image,
        pts=[contour],
        color=255,
        offset=(
            -bounding_box_in_frame_coordinates[0][0],
            -bounding_box_in_frame_coordinates[0][1],
        ),
    )

    return temp_image


def compute_max_distance_transform(blob: Blob):
    temp_image = generate_temp_image(  # TODO there's a Blob.method for that
        blob.contour, blob.bounding_box_in_frame_coordinates
    )
    return np.max(
        cv2.distanceTransform(temp_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    )


def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def get_eroded_blobs(video, blobs_in_frame, frame_number):
    # logging.debug('Getting eroded blobs')
    segmented_frame = np.zeros((video.height, video.width), np.uint8)

    for blob in blobs_in_frame:
        segmented_frame = cv2.fillPoly(segmented_frame, blob.contour, 255)

    segmented_eroded_frame = erode(segmented_frame, video.erosion_kernel_size)

    # Extract blobs info
    contours = cv2.findContours(
        segmented_eroded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    # boundingBoxes, _, centroids, _, pixels_all, contours, _ = blob_extractor(
    #     segmented_eroded_frame, segmented_eroded_frame, (0, np.inf)
    # )
    # logging.debug('Finished getting eroded blobse')
    eroded_blobs_in_frame = []
    for i, contour in enumerate(contours):
        eroded_blob = Blob(
            contour,
            number_of_animals=video.number_of_animals,
            frame_number=frame_number,
            in_frame_index=i,
            video_path=video.video_paths,
            pixels_are_from_eroded_blob=True,
            resolution_reduction=video.resolution_reduction,
        )
        eroded_blobs_in_frame.append(eroded_blob)

    return eroded_blobs_in_frame
