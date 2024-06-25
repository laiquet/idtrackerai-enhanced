from typing import Literal, Sequence

import numpy as np

from idtrackerai import Fragment, ListOfFragments, Session
from idtrackerai.utils import track


def none_max(*args):
    return max(filter(lambda item: item is not None, args))


def get_candidate_identities_by_minimum_speed(
    fragment: Fragment,
    fragments: Sequence[Fragment],
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
    """
    speed_of_candidate_identities: list[float] = []
    current_identity = fragment.identity_corrected_solving_jumps
    for identity in available_identities:
        fragment.identity_corrected_solving_jumps = identity
        neighbour_fragment_past = fragment.get_neighbour_fragment(
            fragments, "to_the_past"
        )
        neighbour_fragment_future = fragment.get_neighbour_fragment(
            fragments, "to_the_future"
        )
        velocity_past = fragment.compute_border_velocity(neighbour_fragment_past)
        velocity_future = fragment.compute_border_velocity(neighbour_fragment_future)

        if velocity_past is None and velocity_future is None:
            speed_of_candidate_identities.append(impossible_velocity_threshold)
        else:
            speed_of_candidate_identities.append(
                none_max(velocity_past, velocity_future)
            )
    fragment.identity_corrected_solving_jumps = current_identity
    argsort_identities_by_speed = np.argsort(speed_of_candidate_identities)
    return (
        np.asarray(available_identities)[argsort_identities_by_speed],
        np.asarray(speed_of_candidate_identities)[argsort_identities_by_speed],
    )


def get_candidate_identities_above_random_P2(
    fragment: Fragment,
    fragments: Sequence[Fragment],
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

    if fragment.n_images == 1:
        random_threshold = 1 / number_of_animals
    else:
        random_threshold = 1 / fragment.n_images
    return (P2_vector > random_threshold).nonzero()[0] + 1


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

    coexisting_identities = {
        coexisting_fragment.assigned_identities[0]
        for coexisting_fragment in fragment.coexisting_individual_fragments
    } - {0, None}

    identities_outside_exclusive_roi: set[int] = set(
        (
            np.argwhere(list_of_fragments.id_to_exclusive_roi != fragment.exclusive_roi)
            + 1
        ).flatten()
    )
    all_identities = set(range(1, list_of_fragments.n_animals + 1))

    non_available_identities = coexisting_identities | identities_outside_exclusive_roi
    available_identities = all_identities - non_available_identities

    assert fragment.assigned_identities[0] not in non_available_identities

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
            list_of_fragments.n_animals,
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
        else:
            assert len(candidate_identities) > 1
            if np.count_nonzero(candidate_speeds == np.min(candidate_speeds)) == 1:
                if candidate_speeds[0] < impossible_velocity_threshold:
                    candidate_id = candidate_identities[0]
                else:
                    candidate_id = 0
            else:
                candidate_id = 0

    fragment.identity_corrected_solving_jumps = candidate_id


def get_fragment_with_same_identity(
    list_of_fragments: ListOfFragments,
    fragment: Fragment,
    direction: Literal["to_the_past", "to_the_future"],
) -> tuple[Fragment | None, int]:
    "Get the `neighbour_fragment` with the same identity in a given `direction`"

    if direction == "to_the_past":
        if fragment.identifier == 0:
            return None, 0

        for neighbour in list_of_fragments.fragments[fragment.identifier - 1 :: -1]:
            if (
                neighbour.is_an_individual
                and neighbour.assigned_identities[0] == fragment.assigned_identities[0]
            ):
                number_of_frames_in_direction = (
                    fragment.start_frame - neighbour.end_frame + 1
                )
                return neighbour, number_of_frames_in_direction

    elif direction == "to_the_future":
        for neighbour in list_of_fragments.fragments[fragment.identifier + 1 :]:
            if (
                neighbour.is_an_individual
                and neighbour.assigned_identities[0] == fragment.assigned_identities[0]
            ):
                number_of_frames_in_direction = (
                    neighbour.start_frame - fragment.end_frame + 1
                )
                return neighbour, number_of_frames_in_direction
    else:
        raise ValueError(direction)
    return None, 0


def compute_neighbour_fragments_and_velocities(
    list_of_fragments: ListOfFragments, fragment: Fragment
) -> tuple[Fragment | None, Fragment | None, float | None, float | None]:
    """Computes the fragments with the same identities to the past and to the
    future of a given `fragment` and gives the velocities at the extremes of
    the current `fragment`

    Parameters
    ----------
    session : <Session object>
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
    """
    neighbour_fragment_past, n_frames_in_past = get_fragment_with_same_identity(
        list_of_fragments, fragment, "to_the_past"
    )
    neighbour_fragment_future, n_frames_in_future = get_fragment_with_same_identity(
        list_of_fragments, fragment, "to_the_future"
    )

    velocity_past = fragment.compute_border_velocity(neighbour_fragment_past)
    velocity_future = fragment.compute_border_velocity(neighbour_fragment_future)

    return (
        neighbour_fragment_past,
        neighbour_fragment_future,
        velocity_past / n_frames_in_past if velocity_past is not None else None,
        velocity_future / n_frames_in_future if velocity_future is not None else None,
    )


def correct_impossible_velocity_jumps_loop(
    session: Session,
    list_of_fragments: ListOfFragments,
    scope: Literal["to_the_past", "to_the_future"],
):
    """Checks whether the velocity needed to join two consecutive fragments with
    the same identity is consistent with the typical velocity of the animals in
    the video (`video.velocity_threshold`). If the velocity is not consistent the
    identity of one of the fragments is reassigned. The check is performed from the
    `video.first_frame_first_global_fragment` to the past or to the future according
    to the `scope`

    Parameters
    ----------
    session : <Session object>
        Object collecting all the parameters of the video and paths for saving
        and loading
    list_of_fragments : <ListOfFragments object>
        Object collecting the list of fragments and all the statistics and methods
        related to them
    scope : string
        If `scope` = `to_the_past` the check is performed to the past and if
        `scope` = `to_the_future` the check is performed to the future.
    """
    fragments_in_direction = list_of_fragments.get_ordered_list_of_fragments(
        scope, session.first_frame_first_global_fragment[session.accumulation_trial]
    )
    velocity_threshold = session.velocity_threshold

    # at this point all individual fragments have a valid identity or 0
    for fragment in track(
        fragments_in_direction, f"Correcting impossible velocity jumps {scope}"
    ):
        # TODO This loops is too slow
        if fragment.is_a_crossing or fragment.assigned_identities[0] == 0:
            continue

        (
            neighbour_fragment_past,
            neighbour_fragment_future,
            velocity_to_past,
            velocity_to_future,
        ) = compute_neighbour_fragments_and_velocities(list_of_fragments, fragment)

        if (
            velocity_to_past is not None
            and velocity_to_future is not None
            and velocity_to_past > velocity_threshold
            and velocity_to_future > velocity_threshold
        ):
            assert neighbour_fragment_future is not None
            assert neighbour_fragment_past is not None
            if (
                neighbour_fragment_past.identity_is_fixed
                or neighbour_fragment_future.identity_is_fixed
            ):
                reassign(fragment, list_of_fragments, velocity_threshold)
            else:
                neighbour_fragment_past_past = (
                    neighbour_fragment_past.get_neighbour_fragment(
                        list_of_fragments.fragments, "to_the_past"
                    )
                )
                velocity_past_past = neighbour_fragment_past.compute_border_velocity(
                    neighbour_fragment_past_past
                )

                neighbour_fragment_future_future = (
                    neighbour_fragment_future.get_neighbour_fragment(
                        list_of_fragments.fragments, "to_the_future"
                    )
                )
                velocity_future_future = (
                    neighbour_fragment_future.compute_border_velocity(
                        neighbour_fragment_future_future
                    )
                )

                if (
                    velocity_past_past is not None
                    and velocity_past_past < velocity_threshold
                ) or (
                    velocity_future_future is not None
                    and velocity_future_future < velocity_threshold
                ):
                    reassign(fragment, list_of_fragments, velocity_threshold)
        elif velocity_to_past is not None and velocity_to_past > velocity_threshold:
            assert neighbour_fragment_past is not None
            if neighbour_fragment_past.identity_is_fixed:
                reassign(fragment, list_of_fragments, velocity_threshold)
            else:
                reassign(neighbour_fragment_past, list_of_fragments, velocity_threshold)
        elif velocity_to_future is not None and velocity_to_future > velocity_threshold:
            assert neighbour_fragment_future is not None
            if neighbour_fragment_future.identity_is_fixed:
                reassign(fragment, list_of_fragments, velocity_threshold)
            else:
                reassign(
                    neighbour_fragment_future, list_of_fragments, velocity_threshold
                )


def correct_impossible_velocity_jumps(
    session: Session, list_of_fragments: ListOfFragments
):
    """Corrects the parts of the video where the velocity of any individual is
    higher than a particular velocity threshold given by `video.velocity_threshold`.
    This check is done from the `video.first_frame_first_global_fragment` to the
    past and to the future

    Parameters
    ----------
    session : <Session object>
        Object collecting all the parameters of the video and paths for saving and loading
    list_of_fragments : <ListOfFragments object>
        Object collecting the list of fragments and all the statistics and methods
        related to them

    See Also
    --------
    correct_impossible_velocity_jumps_loop

    """
    correct_impossible_velocity_jumps_loop(
        session, list_of_fragments, scope="to_the_past"
    )
    correct_impossible_velocity_jumps_loop(
        session, list_of_fragments, scope="to_the_future"
    )
