"""The correct_impossible_velocity_jumps module"""
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

import numpy as np

from idtrackerai import Fragment, ListOfFragments, Video
from idtrackerai.utils import track


def get_candidate_identities_by_minimum_speed(
    fragment: Fragment,
    fragments: list[Fragment],
    available_identities: list[int],
    impossible_velocity_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the candidate identities for a given `fragment` taking into
    consideration the velocities needed to join the `fragment` with its neighbour
    fragments in the past and in the future

    Parameters
    ----------
    fragment : <Fragment object>
        Object collecting all the information for a consecutive set of overlapping
        blobs that are considered to be the same animal
    fragments : list
        List with all the `Fragment` objects of the video
    available_identities : set
        Set with the available idenities for the `fragment`
    impossible_velocity_threshold : float
        If the velocity needed to link two fragments is higher than this threshold
        the identiy of one of the fragments is considerd to be wrong as it would be
        physically impossible for an animal to move so much. See `video.velocity_threshold`
        for each definition

    Returns
    -------
    candidate_identities_by_speed : nd.array
        Array with the identities that fullfill the `impossible_velocity_threshold`
        ordered from minimum to maximum velocity

    speed_of_candidate_identities : nd.array
        Array with the maximum velocity needed to link the given `fragment`
        with its neighbours assuming a given identity. Ordered from minimum to maximum
        velocity

    See Also
    --------
    Fragment
    compute_velocities_consecutive_fragments

    """
    speed_of_candidate_identities: list[float] = []
    for identity in available_identities:
        fragment.user_generated_identity = identity
        neighbour_fragment_past = fragment.get_neighbour_fragment(
            fragments, "to_the_past"
        )
        neighbour_fragment_future = fragment.get_neighbour_fragment(
            fragments, "to_the_future"
        )
        velocities_between_fragments = compute_velocities_consecutive_fragments(
            neighbour_fragment_past, fragment, neighbour_fragment_future
        )

        if np.isnan(velocities_between_fragments).all():
            speed_of_candidate_identities.append(impossible_velocity_threshold)
        else:
            speed_of_candidate_identities.append(
                np.nanmax(velocities_between_fragments)
            )
    fragment.user_generated_identity = None
    argsort_identities_by_speed = np.argsort(speed_of_candidate_identities)
    return (
        np.asarray(available_identities)[argsort_identities_by_speed],
        np.asarray(speed_of_candidate_identities)[argsort_identities_by_speed],
    )


def get_candidate_identities_above_random_P2(
    fragment: Fragment,
    fragments: list[Fragment],
    non_available_identities: np.ndarray,
    available_identities: list[int],
    impossible_velocity_threshold: float,
    number_of_animals: int,
):
    """Computes the candidate identities of a `fragment` taking into
    consideration the probability of identification given by its
    `fragment.P2_vector`. An identity is a potential candidate if the
    probability of identification is above random.

    Parameters
    ----------
    fragment : <Fragment object>
        Object collecting all the information for a consecutive set of
        overlapping blobs that are considered to be the same animal
    fragments : list
        List with all the `Fragment` objects of the video
    non_available_identities : nd.array
        Array with the non available identities for the `fragment`
    available_identities : set
        Set with the available idenities for the `fragment`
    impossible_velocity_threshold : float
        If the velocity needed to link two fragments is higher than this
        threshold the identiy of one of the fragments is considerd to be
        wrong as it would be physically impossible for an animal to move so
        much. See `video.velocity_threshold` for each definition

    Returns
    -------
    candidate_identities_by_speed : nd.array
        Array with the identities that fullfill the
        `impossible_velocity_threshold` ordered from minimum to maximum
        velocity

    speed_of_candidate_identities : nd.array
        Array with the maximum velocity needed to link the given `fragment`
        with its neighbours assuming a given identity. Ordered from
        minimum to maximum velocity

    See Also
    --------
    Fragment
    get_candidate_identities_by_minimum_speed

    """
    P2_vector = fragment.P2_vector
    assert P2_vector is not None
    if len(non_available_identities) > 0:
        P2_vector[non_available_identities - 1] = 0
    if all(P2_vector == 0):
        candidate_identities_speed, _ = get_candidate_identities_by_minimum_speed(
            fragment, fragments, available_identities, impossible_velocity_threshold
        )
        return candidate_identities_speed

    if fragment.number_of_images == 1:
        random_threshold = 1 / number_of_animals
    else:
        random_threshold = 1 / fragment.number_of_images
    return np.argwhere(P2_vector > random_threshold)[:, 0] + 1


def reassign(
    fragment: Fragment,
    list_of_fragments: ListOfFragments,
    impossible_velocity_threshold: float,
):
    """Reassigns the identity of a given `fragment` considering the identity of the
    `fragments` coexisting with it and the `impossible_velocity_threshold`

    Parameters
    ----------
    fragment : <Fragment object>
        Object collecting all the information for a consecutive set of overlapping
        blobs that are considered to be the same animal
    fragments : list
        List with all the `Fragment` objects of the video
    impossible_velocity_threshold : float
        If the velocity needed to link two fragments is higher than this threshold
        the identiy of one of the fragments is considerd to be wrong as it would be
        physically impossible for an animal to move so much. See `video.velocity_threshold`
        for each definition

    See Also
    --------
    :class:`fragment.Fragment`
    :meth:`get_available_and_non_available_identities`
    :meth:`get_candidate_identities_by_minimum_speed`
    :meth:`get_candidate_identities_above_random_P2`

    """

    non_available_identities = {
        coexisting_fragment.assigned_identities[0]
        for coexisting_fragment in fragment.coexisting_individual_fragments
    }
    available_identities = (
        set(range(1, list_of_fragments.number_of_animals + 1))
        - non_available_identities
    )
    if fragment.assigned_identities[0] not in (None, 0):
        available_identities.add(fragment.assigned_identities[0])

    if 0 in non_available_identities:
        non_available_identities.remove(0)
    non_available_identities = np.asarray(list(non_available_identities))

    if len(available_identities) == 1:
        candidate_id = available_identities.pop()
    else:
        available_identities = list(available_identities)
        candidate_identities_speed, speed_of_candidate_identities = (
            get_candidate_identities_by_minimum_speed(
                fragment,
                list_of_fragments.fragments,
                available_identities,
                impossible_velocity_threshold,
            )
        )
        candidate_identities_P2 = get_candidate_identities_above_random_P2(
            fragment,
            list_of_fragments.fragments,
            non_available_identities,
            available_identities,
            impossible_velocity_threshold,
            list_of_fragments.number_of_animals,
        )
        candidate_identities: list[int] = []
        candidate_speeds: list[float] = []
        for candidate_id, candidate_speed in zip(
            candidate_identities_speed, speed_of_candidate_identities
        ):
            if candidate_id in candidate_identities_P2:
                candidate_identities.append(candidate_id)
                candidate_speeds.append(candidate_speed)
        if not candidate_identities:
            candidate_id = 0
        elif len(candidate_identities) == 1:
            if candidate_speeds[0] < impossible_velocity_threshold:
                candidate_id = candidate_identities[0]
            else:
                candidate_id = 0
        elif len(candidate_identities) > 1:
            if np.count_nonzero(candidate_speeds == np.min(candidate_speeds)) == 1:
                if candidate_speeds[0] < impossible_velocity_threshold:
                    candidate_id = candidate_identities[0]
                else:
                    candidate_id = 0
            else:
                candidate_id = 0

    fragment.identity_corrected_solving_jumps = candidate_id


def compute_velocities_consecutive_fragments(
    neighbour_fragment_past: Fragment | None,
    fragment: Fragment,
    neighbour_fragment_future: Fragment | None,
) -> list[float]:
    """Compute velocities in the extremes of a `fragment` with respecto to its
    `neighbour_fragment_past` and `neighbour_fragment_future`

    Parameters
    ----------
    neighbour_fragment_past : <Fragment object>
        `Fragment` object with the same identity as the current fragment in the
        past
    fragment : <Fragment object>
        Object collecting all the information for a consecutive set of
        overlapping blobs that are considered to be the same animal
    neighbour_fragment_future : <Fragment object>
        `Fragment` object with the same identity as the current fragment in the
        future

    Returns
    -------
    velocities : list
        List with the velocity needed to link the fragment with its fragment in
        the past and in the future

    See Also
    --------
    Fragment

    """
    velocities = [np.nan, np.nan]
    if neighbour_fragment_past is not None:
        velocities[0] = fragment.compute_border_velocity(neighbour_fragment_past)
    if neighbour_fragment_future is not None:
        velocities[1] = neighbour_fragment_future.compute_border_velocity(fragment)
    return velocities


def get_fragment_with_same_identity(
    number_of_frames: int,
    list_of_fragments: ListOfFragments,
    fragment: Fragment,
    direction: str,
) -> tuple[Fragment | None, int]:
    """Get the `neighbour_fragment` with the same identity in a given `direction`

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving
        and loading
    list_of_fragments : <ListOfFragments object>
        Object collecting the list of fragments and all the statistics and
        methods related to them
    fragment : <Fragment object>
        Object collecting all the information for a consecutive set of
        overlapping blobs that are considered to be the same animal
    direction : string
        If `direction` = `to_the_past` gets the `neighbour_fragment` in the
        past, `direction` = `to_the_future` gets the `neighbour_fragment` in
        the future

    Returns
    -------
    neighbour_fragment : <Fragment object>
        `Fragment` object with the same identity in a given `direction`
    number_of_frames_in_direction : int
        Number of frames to find the `neighbour_fragment` from a given extreme
        of the `fragment`

    See Also
    --------
    Fragment

    """
    number_of_frames_in_direction = 0
    frame_number = (
        fragment.start_frame if direction == "to_the_past" else fragment.end_frame
    )

    neighbour_fragment = None
    while neighbour_fragment is None and 0 < frame_number < number_of_frames:
        neighbour_fragment = fragment.get_neighbour_fragment(
            list_of_fragments.fragments,
            direction,
            number_of_frames_in_direction=number_of_frames_in_direction,
        )
        number_of_frames_in_direction += 1
        frame_number += -1 if direction == "to_the_past" else 1

    return neighbour_fragment, number_of_frames_in_direction


def compute_neighbour_fragments_and_velocities(
    number_of_frames: int, list_of_fragments: ListOfFragments, fragment: Fragment
) -> tuple[Fragment | None, Fragment | None, np.ndarray]:
    """Computes the fragments with the same identities to the past and to the
    future of a given `fragment` and gives the velocities at the extremes of
    the current `fragment`

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving
        and loading
    list_of_fragments : <ListOfFragments object>
        Object collecting the list of fragments and all the statistics and
        methods related to them
    fragment : <Fragment object>
        Object collecting all the information for a consecutive set of
        overlapping blobs that are considered to be the same animal

    Returns
    -------
    neighbour_fragment_past : <Fragment object>
        `Fragment` object with the same identity as the current fragment in the
        past
    neighbour_fragment_future : <Fragment object>
        `Fragment` object with the same identity as the current fragment in the
        future
    velocities_between_fragments : nd.array
        Velocities needed to connect the current fragment to its consecutive
        fragments in the past and in the future.

    See Also
    --------
    get_fragment_with_same_identity
    compute_velocities_consecutive_fragments

    """
    neighbour_fragment_past, n_frames_in_past = get_fragment_with_same_identity(
        number_of_frames, list_of_fragments, fragment, "to_the_past"
    )
    neighbour_fragment_future, n_frames_in_future = get_fragment_with_same_identity(
        number_of_frames, list_of_fragments, fragment, "to_the_future"
    )

    velocities = compute_velocities_consecutive_fragments(
        neighbour_fragment_past, fragment, neighbour_fragment_future
    )
    velocities_between_fragments = np.asarray(velocities) / np.asarray(
        [n_frames_in_past, n_frames_in_future]
    )

    return (
        neighbour_fragment_past,
        neighbour_fragment_future,
        velocities_between_fragments,
    )


def correct_impossible_velocity_jumps_loop(
    video: Video, list_of_fragments: ListOfFragments, scope: str
):
    """Checks whether the velocity needed to join two consecutive fragments with
    the same identity is consistent with the typical velocity of the animals in
    the video (`video.velocity_threshold`). If the velocity is not consistent the
    identity of one of the fragments is reassigned. The check is performed from the
    `video.first_frame_first_global_fragment` to the past or to the future according
    to the `scope`

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving
        and loading
    list_of_fragments : <ListOfFragments object>
        Object collecting the list of fragments and all the statistics and methods
        related to them
    scope : string
        If `scope` = `to_the_past` the check is performed to the past and if
        `scope` = `to_the_future` the check is performed to the future.

    See Also
    --------
    Fragment
    compute_neighbour_fragments_and_velocities
    compute_velocities_consecutive_fragments
    reassign

    """
    fragments_in_direction = list_of_fragments.get_ordered_list_of_fragments(
        scope, video.first_frame_first_global_fragment[video.accumulation_trial]
    )
    impossible_velocity_threshold = video.velocity_threshold

    for fragment in track(
        fragments_in_direction, f"Correcting impossible velocity jumps {scope}"
    ):
        # TODO This loops is too slow
        if fragment.is_a_crossing or fragment.assigned_identities[0] == 0:
            continue

        (
            neighbour_fragment_past,
            neighbour_fragment_future,
            velocities_between_fragments,
        ) = compute_neighbour_fragments_and_velocities(
            video.number_of_frames, list_of_fragments, fragment
        )

        if all(velocities_between_fragments > impossible_velocity_threshold):
            assert neighbour_fragment_future is not None
            assert neighbour_fragment_past is not None
            if (
                neighbour_fragment_past.identity_is_fixed
                or neighbour_fragment_future.identity_is_fixed
            ):
                reassign(fragment, list_of_fragments, impossible_velocity_threshold)
            else:
                neighbour_fragment_past_past = (
                    neighbour_fragment_past.get_neighbour_fragment(
                        list_of_fragments.fragments, "to_the_past"
                    )
                )
                velocity_in_past = compute_velocities_consecutive_fragments(
                    neighbour_fragment_past_past, neighbour_fragment_past, fragment
                )[0]
                neighbour_fragment_future_future = (
                    neighbour_fragment_future.get_neighbour_fragment(
                        list_of_fragments.fragments, "to_the_future"
                    )
                )
                velocity_in_future = compute_velocities_consecutive_fragments(
                    fragment,
                    neighbour_fragment_future,
                    neighbour_fragment_future_future,
                )[1]
                if (
                    velocity_in_past < impossible_velocity_threshold
                    or velocity_in_future < impossible_velocity_threshold
                ):
                    reassign(fragment, list_of_fragments, impossible_velocity_threshold)
        elif velocities_between_fragments[0] > impossible_velocity_threshold:
            assert neighbour_fragment_past is not None
            if neighbour_fragment_past.identity_is_fixed:
                reassign(fragment, list_of_fragments, impossible_velocity_threshold)
            else:
                reassign(
                    neighbour_fragment_past,
                    list_of_fragments,
                    impossible_velocity_threshold,
                )
        elif velocities_between_fragments[1] > impossible_velocity_threshold:
            assert neighbour_fragment_future is not None
            if neighbour_fragment_future.identity_is_fixed:
                reassign(fragment, list_of_fragments, impossible_velocity_threshold)
            else:
                reassign(
                    neighbour_fragment_future,
                    list_of_fragments,
                    impossible_velocity_threshold,
                )


def correct_impossible_velocity_jumps(video: Video, list_of_fragments: ListOfFragments):
    """Corrects the parts of the video where the velocity of any individual is
    higher than a particular velocity threshold given by `video.velocity_threshold`.
    This check is done from the `video.first_frame_first_global_fragment` to the
    past and to the future

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    list_of_fragments : <ListOfFragments object>
        Object collecting the list of fragments and all the statistics and methods
        related to them

    See Also
    --------
    correct_impossible_velocity_jumps_loop

    """
    correct_impossible_velocity_jumps_loop(
        video, list_of_fragments, scope="to_the_past"
    )
    correct_impossible_velocity_jumps_loop(
        video, list_of_fragments, scope="to_the_future"
    )
