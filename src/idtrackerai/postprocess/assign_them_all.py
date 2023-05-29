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
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from idtrackerai import Blob, ListOfBlobs, ListOfFragments, Video
from idtrackerai.utils import track

from .compute_velocity_model import compute_model_velocity
from .erosion import compute_erosion_disk, get_eroded_blobs


def set_individual_with_identity_0_as_crossings(list_of_fragments: ListOfFragments):
    for fragment in list_of_fragments.individual_fragments:
        if (
            len(fragment.assigned_identities) == 1
            and fragment.assigned_identities[0] == 0
        ):
            fragment.is_an_individual = False
            fragment.forced_crossing = True
            fragment.identity = None
            fragment.identity_corrected_solving_jumps = None


def find_the_gap_interval(
    blobs_in_video: list[list[Blob]],
    possible_identities: set[int],
    gap_start: int,
    list_of_occluded_identities: list[set[int]],
):
    there_are_missing_identities = True
    frame_number = gap_start + 1
    if frame_number < len(blobs_in_video):
        while there_are_missing_identities and 0 < frame_number < len(blobs_in_video):
            blobs_in_frame = blobs_in_video[frame_number]
            occluded_identities_in_frame = list_of_occluded_identities[frame_number]
            missing_identities = get_missing_identities_from_blobs_in_frame(
                possible_identities, blobs_in_frame, occluded_identities_in_frame
            )
            if not missing_identities or frame_number == len(blobs_in_video) - 1:
                there_are_missing_identities = False
            else:
                frame_number += 1
            gap_end = frame_number
    else:
        gap_end = gap_start
    return gap_start, gap_end


def get_blob_by_identity(blobs_in_frame: list[Blob], identity: int) -> Blob | None:
    for blob in blobs_in_frame:
        if identity in blob.final_identities:
            return blob
    return None


def get_candidate_blobs_by_overlapping(
    blob_to_test: Blob, eroded_blobs_in_frame: list[Blob]
) -> list[Blob]:
    overlapping_blobs = [
        blob for blob in eroded_blobs_in_frame if blob_to_test.overlaps_with(blob)
    ]
    return overlapping_blobs if overlapping_blobs else eroded_blobs_in_frame


def get_missing_identities_from_blobs_in_frame(
    possible_identities: set[int],
    blobs_in_frame: list[Blob],
    occluded_identities_in_frame: set[int],
) -> set:
    identities_in_frame = set()
    for blob in blobs_in_frame:
        identities_in_frame.update(blob.final_identities)
    return (possible_identities - identities_in_frame) - occluded_identities_in_frame


def get_candidate_centroid(
    individual_gap_interval: tuple[int, int],
    previous_blob_to_the_gap: Blob,
    next_blob_to_the_gap: Blob,
    identity: int,
    border: str,
) -> tuple[float, float]:
    blobs_for_interpolation = [previous_blob_to_the_gap, next_blob_to_the_gap]
    centroids_to_interpolate = [
        list(blob.final_centroids)[list(blob.final_identities).index(identity)]
        for blob in blobs_for_interpolation
    ]
    centroids_to_interpolate = np.asarray(centroids_to_interpolate).T
    argsort_x = np.argsort(centroids_to_interpolate[0])
    centroids_to_interpolate[0] = centroids_to_interpolate[0][argsort_x]
    centroids_to_interpolate[1] = centroids_to_interpolate[1][argsort_x]
    number_of_points = individual_gap_interval[1] - individual_gap_interval[0] + 1
    x_interp = np.linspace(
        centroids_to_interpolate[0][0],
        centroids_to_interpolate[0][1],
        number_of_points + 1,
    )
    y_interp = np.interp(
        x_interp, centroids_to_interpolate[0], centroids_to_interpolate[1]
    )
    if border == "start" and all(argsort_x == np.asarray([0, 1])):
        return list(zip(x_interp, y_interp))[1]
    if border == "start" and all(argsort_x == np.asarray([1, 0])):
        return list(zip(x_interp, y_interp))[-2]
    if border == "end" and all(argsort_x == np.asarray([0, 1])):
        return list(zip(x_interp, y_interp))[-2]
    if border == "end" and all(argsort_x == np.asarray([1, 0])):
        return list(zip(x_interp, y_interp))[1]
    raise ValueError(border, argsort_x)


def find_the_individual_gap_interval(
    blobs_in_video: list[list[Blob]],
    possible_identities: set[int],
    identity: int,
    a_frame_in_the_gap: int,
    list_of_occluded_identities: list[set[int]],
) -> tuple[int, int]:
    # logging.debug('Finding the individual gap interval')
    # find gap start
    identity_is_missing = True
    gap_start = a_frame_in_the_gap
    frame_number = gap_start

    # logging.debug('To the past while loop')
    while identity_is_missing and 0 < frame_number < len(blobs_in_video):
        blobs_in_frame = blobs_in_video[frame_number]
        occluded_identities_in_frame = list_of_occluded_identities[frame_number]
        missing_identities = get_missing_identities_from_blobs_in_frame(
            possible_identities, blobs_in_frame, occluded_identities_in_frame
        )
        if identity not in missing_identities:
            gap_start = frame_number + 1
            identity_is_missing = False
        else:
            frame_number -= 1

    # find gap end
    identity_is_missing = True
    frame_number = a_frame_in_the_gap
    gap_end = a_frame_in_the_gap

    # logging.debug('To the future while loop')
    while identity_is_missing and 0 < frame_number < len(blobs_in_video):
        blobs_in_frame = blobs_in_video[frame_number]
        occluded_identities_in_frame = list_of_occluded_identities[frame_number]
        missing_identities = get_missing_identities_from_blobs_in_frame(
            possible_identities, blobs_in_frame, occluded_identities_in_frame
        )
        if identity not in missing_identities:
            gap_end = frame_number
            identity_is_missing = False
        else:
            gap_end += 1
            frame_number = gap_end
    # logging.debug('Finished finding the individual gap interval')
    return gap_start, gap_end


def get_previous_and_next_blob_wrt_gap(
    blobs_in_video: list[list[Blob]],
    possible_identities: set[int],
    identity: int,
    frame_number: int,
    list_of_occluded_identities: list[set[int]],
) -> tuple[tuple[int, int], Blob | None, Blob | None]:
    # logging.debug('Finding previons and next blobs to the gap of this identity')
    individual_gap_interval = find_the_individual_gap_interval(
        blobs_in_video,
        possible_identities,
        identity,
        frame_number,
        list_of_occluded_identities,
    )
    if individual_gap_interval[0] != 0:
        previous_blob_to_the_gap = get_blob_by_identity(
            blobs_in_video[individual_gap_interval[0] - 1], identity
        )
    else:
        previous_blob_to_the_gap = None

    if individual_gap_interval[1] != len(blobs_in_video):
        next_blob_to_the_gap = get_blob_by_identity(
            blobs_in_video[individual_gap_interval[1]], identity
        )
    else:
        next_blob_to_the_gap = None

    return individual_gap_interval, previous_blob_to_the_gap, next_blob_to_the_gap


def get_closest_contour_point_to(
    contour: np.ndarray, candidate_centroid: tuple[float, float]
):
    return tuple(contour[np.argmin(cdist([candidate_centroid], contour))])


def get_nearest_eroded_blob_to_candidate_centroid(
    eroded_blobs: list[Blob], candidate_centroid: tuple[float, float]
) -> Blob:
    return min(
        eroded_blobs, key=lambda b: b.distance_from_countour_to(candidate_centroid)
    )


def nearest_candidate_blob_is_near_enough(
    velocity_threshold: float,
    candidate_blob: Blob,
    candidate_centroid: tuple[float, float],
    blob_in_border_frame: Blob,
) -> bool:
    points = [candidate_centroid, blob_in_border_frame.centroid]
    return any(
        candidate_blob.distance_to(point) < velocity_threshold for point in points
    )


def centroid_is_inside_of_any_eroded_blob(
    candidate_eroded_blobs: list[Blob], candidate_centroid: tuple[float, float]
) -> list[Blob]:
    # logging.debug('Checking whether the centroids is inside of a blob')
    candidate_centroid = tuple(map(int, candidate_centroid))
    # logging.debug('Finished whether the centroids is inside of a blob')
    return [
        b
        for b in candidate_eroded_blobs
        if cv2.pointPolygonTest(b.contour, candidate_centroid, False) >= 0
    ]


def evaluate_candidate_blobs_and_centroid(
    velocity_threshold: float,
    candidate_eroded_blobs: list[Blob],
    candidate_centroid: tuple[float, float],
    blob_in_border_frame: Blob,
):
    blob_containing_candidate_centroid = centroid_is_inside_of_any_eroded_blob(
        candidate_eroded_blobs, candidate_centroid
    )
    if blob_containing_candidate_centroid:
        return blob_containing_candidate_centroid[0], candidate_centroid
    if candidate_eroded_blobs:
        nearest_blob = get_nearest_eroded_blob_to_candidate_centroid(
            candidate_eroded_blobs, candidate_centroid
        )
        # TODO why the nearest contour point is the new centroid?
        new_centroid = get_closest_contour_point_to(
            nearest_blob.contour, candidate_centroid
        )
        if nearest_candidate_blob_is_near_enough(
            velocity_threshold, nearest_blob, candidate_centroid, blob_in_border_frame
        ) or nearest_blob.overlaps_with(blob_in_border_frame):
            # the candidate centroid is near to a candidate blob
            return nearest_blob, new_centroid
        # the candidate centroids is far from a candidate blob
        return None, None
    # there where no candidate blobs
    return None, None


def get_candidate_tuples_with_centroids_in_original_blob(
    original_blob: Blob,
    candidate_tuples_to_close_gap: list[tuple[Blob, tuple[float, float], int]],
):
    return [
        candidate_tuple
        for candidate_tuple in candidate_tuples_to_close_gap
        if cv2.pointPolygonTest(
            original_blob.contour, tuple(map(int, candidate_tuple[1])), False
        )
        >= 0
    ]


def assign_identity_to_new_blobs(
    blobs_in_video: list[list[Blob]],
    original_inner_blobs_in_frame: list[Blob],
    candidate_tuples_to_close_gap: list[tuple[Blob, tuple[float, float], int]],
    list_of_occluded_identities: list[set[int]],
):
    new_original_blobs: list[Blob] = []

    for original_blob in original_inner_blobs_in_frame:
        candidate_tuples_with_centroids_in_original_blob = (
            get_candidate_tuples_with_centroids_in_original_blob(
                original_blob, candidate_tuples_to_close_gap
            )
        )
        if len(candidate_tuples_with_centroids_in_original_blob) == 1:
            # the gap is a single individual blob
            identity = candidate_tuples_with_centroids_in_original_blob[0][2]
            centroid = candidate_tuples_with_centroids_in_original_blob[0][1]
            original_blob_final_identities = list(original_blob.final_identities)
            if (
                original_blob.is_an_individual
                and len(original_blob_final_identities) == 1
                and original_blob_final_identities[0] == 0
            ):
                original_blob.identities_corrected_closing_gaps = [identity]
                # TODO loop over the fragment
                for blobs_in_frame in blobs_in_video:
                    for blob in blobs_in_frame:
                        if (
                            blob.fragment_identifier
                            == original_blob.fragment_identifier
                        ):
                            blob.identities_corrected_closing_gaps = [identity]

            elif original_blob.is_an_individual:
                list_of_occluded_identities[original_blob.frame_number].add(identity)
            elif original_blob.is_a_crossing:
                if original_blob.identities_corrected_closing_gaps is not None:
                    identity = original_blob.identities_corrected_closing_gaps + [
                        identity
                    ]
                    assert original_blob.interpolated_centroids is not None
                    centroid = original_blob.interpolated_centroids + [centroid]
                else:
                    identity = [identity]
                    centroid = [centroid]
                frame_number = original_blob.frame_number
                new_blob = candidate_tuples_with_centroids_in_original_blob[0][0]
                new_blob.frame_number = frame_number
                new_blob.identities_corrected_closing_gaps = identity
                new_blob.interpolated_centroids = centroid
                original_blob = new_blob

            new_original_blobs.append(original_blob)
        elif (
            len(candidate_tuples_with_centroids_in_original_blob) > 1
            and original_blob.is_a_crossing
        ):
            # Note that the original blobs that were unidentified (identity 0)
            # are set to zero before starting the main while loop
            # logging.debug('Many candidate tuples for this original blob, '
            #              'and the original blob is a crossing')
            # HERE
            candidate_eroded_blobs: list[Blob] = []
            candidate_eroded_blobs_centroids: list[tuple[float, float]] = []
            candidate_eroded_blobs_identities: list[int] = []
            for blob, centroid, id in candidate_tuples_with_centroids_in_original_blob:
                candidate_eroded_blobs.append(blob)
                candidate_eroded_blobs_centroids.append(centroid)
                candidate_eroded_blobs_identities.append(id)

            if len(set(candidate_eroded_blobs)) == 1:  # crossing not split
                original_blob.interpolated_centroids = candidate_eroded_blobs_centroids
                original_blob.identities_corrected_closing_gaps = (
                    candidate_eroded_blobs_identities
                )

                original_blob.contour = candidate_eroded_blobs[0].contour
                new_original_blobs.append(original_blob)

            elif len(set(candidate_eroded_blobs)) > 1:  # crossing split
                count_eroded_blobs = {
                    eroded_blob: candidate_eroded_blobs.count(eroded_blob)
                    for eroded_blob in candidate_eroded_blobs
                }
                for (
                    eroded_blob,
                    centroid,
                    identity,
                ) in candidate_tuples_with_centroids_in_original_blob:
                    if count_eroded_blobs[eroded_blob] == 1:
                        # split blob, single individual
                        eroded_blob.frame_number = original_blob.frame_number
                        eroded_blob.centroid = centroid
                        eroded_blob.identities_corrected_closing_gaps = [identity]
                        eroded_blob.is_an_individual = True
                        eroded_blob.forced_crossing = True
                        eroded_blob.was_a_crossing = True
                        new_original_blobs.append(eroded_blob)
                    elif count_eroded_blobs[eroded_blob] > 1:
                        if eroded_blob.interpolated_centroids is None:
                            eroded_blob.interpolated_centroids = []
                        if eroded_blob.identities_corrected_closing_gaps is None:
                            eroded_blob.identities_corrected_closing_gaps = []
                        eroded_blob.frame_number = original_blob.frame_number
                        eroded_blob.interpolated_centroids.append(centroid)
                        eroded_blob.identities_corrected_closing_gaps.append(identity)
                        eroded_blob.is_an_individual = False
                        eroded_blob.forced_crossing = True
                        new_original_blobs.append(eroded_blob)

        new_original_blobs.append(original_blob)

    new_original_blobs = list(set(new_original_blobs))
    blobs_in_video[original_blob.frame_number] = new_original_blobs
    return blobs_in_video, list_of_occluded_identities


def get_forward_backward_list_of_frames(gap_interval: tuple[int, int]):
    """input:
    gap_interval: array of tuple [start_frame_number, end_frame_number]
    output:
    [f1, fn, f2, fn-1, ...] for f1 = start_frame_number and
                                fn = end_frame_number"""
    # logging.debug('Got forward-backward list of frames')
    gap_range = range(*gap_interval)
    gap_length = len(gap_range)
    return np.insert(gap_range[::-1], np.arange(gap_length), gap_range)[:gap_length]


def interpolate_trajectories_during_gaps(
    video: Video,
    blobs_in_video: list[list[Blob]],
    list_of_occluded_identities: list[set[int]],
    possible_identities: set[int],
    erosion_counter: int,
):
    for frame_number in track(
        range(1, video.number_of_frames), f"Closing gaps, iteration {erosion_counter}"
    ):
        blobs_in_frame = blobs_in_video[frame_number]
        occluded_identities_in_frame = list_of_occluded_identities[frame_number]

        missing_identities = get_missing_identities_from_blobs_in_frame(
            possible_identities, blobs_in_frame, occluded_identities_in_frame
        )
        if not missing_identities or not blobs_in_frame:
            continue

        gap_interval = find_the_gap_interval(
            blobs_in_video,
            possible_identities,
            frame_number,
            list_of_occluded_identities,
        )
        forward_backward_list_of_frames = get_forward_backward_list_of_frames(
            gap_interval
        )
        for index, inner_frame_number in enumerate(forward_backward_list_of_frames):
            inner_occluded_identities_in_frame = list_of_occluded_identities[
                inner_frame_number
            ]
            inner_blobs_in_frame = blobs_in_video[inner_frame_number]
            if not inner_blobs_in_frame:
                continue

            if erosion_counter == 0:
                eroded_blobs_in_frame = inner_blobs_in_frame
            else:
                eroded_blobs_in_frame = get_eroded_blobs(
                    video, inner_blobs_in_frame, inner_frame_number
                )
                if not eroded_blobs_in_frame:
                    eroded_blobs_in_frame = inner_blobs_in_frame

            inner_missing_identities = get_missing_identities_from_blobs_in_frame(
                possible_identities,
                inner_blobs_in_frame,
                inner_occluded_identities_in_frame,
            )
            candidate_tuples_to_close_gap: list[
                tuple[Blob, tuple[float, float], int]
            ] = []
            for identity in inner_missing_identities:
                (
                    individual_gap_interval,
                    previous_blob_to_the_gap,
                    next_blob_to_the_gap,
                ) = get_previous_and_next_blob_wrt_gap(
                    blobs_in_video,
                    possible_identities,
                    identity,
                    inner_frame_number,
                    list_of_occluded_identities,
                )
                if (
                    previous_blob_to_the_gap is not None
                    and next_blob_to_the_gap is not None
                ):
                    border = "start" if index % 2 == 0 else "end"
                    candidate_centroid = get_candidate_centroid(
                        individual_gap_interval,
                        previous_blob_to_the_gap,
                        next_blob_to_the_gap,
                        identity,
                        border=border,
                    )

                    blob_in_border_frame = (
                        previous_blob_to_the_gap
                        if border == "start"
                        else next_blob_to_the_gap
                    )

                    candidate_eroded_blobs = get_candidate_blobs_by_overlapping(
                        blob_in_border_frame, eroded_blobs_in_frame
                    ) + centroid_is_inside_of_any_eroded_blob(
                        eroded_blobs_in_frame, candidate_centroid
                    )

                    candidate_blob_to_close_gap, centroid = (
                        evaluate_candidate_blobs_and_centroid(
                            video.velocity_threshold,
                            candidate_eroded_blobs,
                            candidate_centroid,
                            blob_in_border_frame,
                        )
                    )
                    if candidate_blob_to_close_gap is not None:
                        assert centroid is not None
                        candidate_tuples_to_close_gap.append(
                            (candidate_blob_to_close_gap, centroid, identity)
                        )
                    else:
                        list_of_occluded_identities[inner_frame_number].add(identity)
                # this manages the case in which identities are
                # missing in the first frame or disappear
                # without appearing anymore,
                else:
                    # and eventual occlusions (an identified blob
                    # does not appear in the previous and/or the
                    # next frame)
                    # logging.debug('------There is not next or not '
                    #              'previous blob to this inner gap:'
                    #              ' it must be occluded')
                    # if previous_blob_to_the_gap is None:
                    #     logging.debug('previous_blob_to_the_gap '
                    #                  'is None')
                    # else:
                    #     logging.debug('previous_blob_to_the_gap '
                    #                  'exists')
                    # if next_blob_to_the_gap:
                    #     logging.debug('next_blob_to_the_gap is None')
                    # else:
                    #     logging.debug('next_blob_to_the_gap exists')
                    for i in range(
                        individual_gap_interval[0], individual_gap_interval[1]
                    ):
                        list_of_occluded_identities[i].add(identity)

            blobs_in_video, list_of_occluded_identities = assign_identity_to_new_blobs(
                blobs_in_video,
                inner_blobs_in_frame,
                candidate_tuples_to_close_gap,
                list_of_occluded_identities,
            )
    return blobs_in_video, list_of_occluded_identities


def reset_blobs_in_video_before_erosion_iteration(all_blobs: Iterable[Blob]):
    """Resets the identity of crossings and individual with multiple identities
    before starting a loop of interpolation

    Parameters
    ----------
    blobs_in_video : list of lists of `Blob` objects
    """
    for blob in all_blobs:
        if blob.is_a_crossing:
            blob.identity = None
        elif blob.is_an_individual and len(list(blob.final_identities)) > 1:
            blob.identities_corrected_closing_gaps = None


def close_trajectories_gaps(
    video: Video, list_of_blobs: ListOfBlobs, list_of_fragments: ListOfFragments
):
    """This is the main function to close the gaps where animals have not been
    identified (labelled with identity 0), are crossing with another animals or
    are occluded or not segmented.

    Parameters
    ----------
    video : <Video object>
        Object containing all the parameters of the video.
    list_of_blobs : <ListOfBlobs object>
        Object with the collection of blobs found during segmentation with associated
        methods. See :class:`list_of_blobs.ListOfBlobs`
    list_of_fragments : <ListOfFragments object>
        Collection of individual and crossing fragments with associated methods.
        See :class:`list_of_fragments.ListOfFragments`

    Returns
    -------
    list_of_blobs : <ListOfBlobs object>
        ListOfBlobs object with the updated blobs and identities that close gaps

    See Also
    --------
    :func:`set_individual_with_identity_0_as_crossings`
    :func:`compute_erosion_disk`
    :func:`compute_model_velocity`
    :func:`reset_blobs_in_video_before_erosion_iteration`
    :func:`interpolate_trajectories_during_gaps`
    :func:`closing_gap_stopping_criteria`
    :func:`clean_individual_blob_before_saving`

    """
    set_individual_with_identity_0_as_crossings(list_of_fragments)
    list_of_fragments.update_blobs(list_of_blobs.all_blobs)

    if not hasattr(video, "erosion_kernel_size"):
        video.erosion_kernel_size = compute_erosion_disk(list_of_blobs.blobs_in_video)
    if not hasattr(video, "velocity_threshold"):
        video.velocity_threshold = compute_model_velocity(list_of_fragments)
    possible_identities = set(range(1, video.number_of_animals + 1))
    list_of_occluded_identities: list[set[int]] = [
        set() for _ in range(video.number_of_frames)
    ]

    previous_number_of_non_split_crossings = list_of_blobs.number_of_crossing_blobs
    erosion_counter = 0
    continue_erosion_protocol = True
    # TODO why erosion_counter==1?
    while continue_erosion_protocol or erosion_counter == 1:
        reset_blobs_in_video_before_erosion_iteration(list_of_blobs.all_blobs)
        list_of_blobs.blobs_in_video, list_of_occluded_identities = (
            interpolate_trajectories_during_gaps(
                video,
                list_of_blobs.blobs_in_video,
                list_of_occluded_identities,
                possible_identities,
                erosion_counter,
            )
        )

        current_number_of_non_split_crossings = list_of_blobs.number_of_crossing_blobs

        continue_erosion_protocol = (
            previous_number_of_non_split_crossings
            > current_number_of_non_split_crossings
        )

        previous_number_of_non_split_crossings = current_number_of_non_split_crossings
        erosion_counter += 1

    for blob in list_of_blobs.all_blobs:
        if blob.is_an_individual and len(list(blob.final_identities)) > 1:
            blob.identities_corrected_closing_gaps = None
    return list_of_blobs
