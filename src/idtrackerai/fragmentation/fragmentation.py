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
from contextlib import suppress

import cv2
import numpy as np

from idtrackerai import Blob, ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Video
from idtrackerai.utils import IdtrackeraiError, track


def fragmentation_API(
    video: Video, list_of_blobs: ListOfBlobs
) -> tuple[ListOfFragments, ListOfGlobalFragments]:
    video.fragmentation_timer.start()

    if video.exclusive_rois:
        set_blobs_ROI(list_of_blobs, video.ROI_mask)

    compute_fragment_identifier(list_of_blobs.blobs_in_video)

    list_of_fragments = ListOfFragments.from_fragmented_blobs(
        list_of_blobs.all_blobs, video.n_animals, video.id_images_file_paths
    )
    logging.info(
        f"{list_of_fragments.number_of_fragments} Fragments in total, "
        f"{list_of_fragments.number_of_individual_fragments} individuals and "
        f"{list_of_fragments.number_of_crossing_fragments} crossings"
    )

    with suppress(AttributeError):
        # clean up exclusive ROI attribute, it is on Fragments now
        for blob in list_of_blobs.all_blobs:
            del blob.exclusive_roi

    list_of_global_fragments = ListOfGlobalFragments.from_fragments(
        list_of_blobs.blobs_in_video, list_of_fragments.fragments, video.n_animals
    )
    list_of_fragments.manage_accumulable_non_accumulable_fragments(
        list_of_global_fragments.global_fragments,
        list_of_global_fragments.non_accumulable_global_fragments,
    )

    video.fragmentation_timer.finish()
    return list_of_fragments, list_of_global_fragments


def compute_fragment_identifier(blobs_in_video: list[list[Blob]]):
    """Associates a unique fragment identifier to individual blobs
    connected with its next and previous blobs.

    Blobs must be connected and classified as individuals or crossings.
    """
    frame_id = 0
    for blobs_in_frame in track(blobs_in_video, "Fragmenting blobs"):
        for blob in blobs_in_frame:
            if blob.fragment_identifier != -1:
                continue

            blob.fragment_identifier = frame_id
            while (
                blob.n_next == 1
                and blob.next[0].n_previous == 1
                and blob.next[0].is_an_individual == blob.is_an_individual
                and blob.next[0].exclusive_roi == blob.exclusive_roi
            ):
                blob = blob.next[0]
                blob.fragment_identifier = frame_id

            frame_id += 1


def set_blobs_ROI(list_of_blobs: ListOfBlobs, mask: np.ndarray | None):
    if mask is None:
        raise IdtrackeraiError(
            "Cannot set exclusive ROIs if there's is not a defined ROI"
        )

    contours = find_exclusive_contours(mask)

    for blob in track(
        list_of_blobs.all_blobs,
        "Finding blob's exclusive ROI",
        list_of_blobs.number_of_blobs,
    ):
        blob.exclusive_roi = find_parent_ROI(blob.centroid, contours)


def find_exclusive_contours(
    img: np.ndarray | None,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    if img is None:
        return []
    all_cnts, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    all_cnts = list(map(np.squeeze, all_cnts))
    n_cnts = len(all_cnts)

    contours: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for index, cnt in enumerate(all_cnts):
        if cnt.size < 4:  # one point contours
            continue
        if hierarchy[index][3] == -1:  # this is a top-level contour
            # check all contours that have the current index as their parent
            holes = [
                all_cnts[i]
                for i in range(n_cnts)
                if hierarchy[i][3] == index and all_cnts[i].size >= 4
            ]
            contours.append((cnt, holes))
    return contours


def find_parent_ROI(
    point: tuple[float, float], contours: list[tuple[np.ndarray, list[np.ndarray]]]
):
    for cont_index, (external, holes) in enumerate(contours):
        if cv2.pointPolygonTest(external, point, False) > 0 and all(
            cv2.pointPolygonTest(hole, point, False) < 0 for hole in holes
        ):
            return cont_index
    raise IdtrackeraiError("Not inside any ROI")
