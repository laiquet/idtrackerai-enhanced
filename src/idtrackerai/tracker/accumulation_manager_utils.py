# This file has been created to temporarily deal with coupling
# between accumulation_manager.py and list_of_global_fragments.py
# TODO remove this file and manage coupling properly
import numpy as np

from idtrackerai import Fragment, GlobalFragment


def get_P1_array_and_argsort(global_fragment: GlobalFragment):
    """Given a global fragment computes P1 for each of its individual
    fragments and returns a
    matrix of sorted indices according to P1

    Parameters
    ----------
    global_fragment : GlobalFragment object
        Collection of images relative to a part of the video in which all
        the animals are visible.

    Returns
    -------
    P1_array : nd.array
        P1 computed for every individual fragment in the global fragment
    index_individual_fragments_sorted_by_P1_max_to_min : nd.array
        Argsort of P1 array of each individual fragment
    """
    # get array of P1 values for the global fragment
    P1_array = np.asarray(
        [fragment.P1_vector for fragment in global_fragment.individual_fragments]
    )
    # get the maximum P1 of each individual fragment
    P1_max = np.max(P1_array, axis=1)
    # logging.debug("P1 max: %s" %str(P1_max))
    # get the index position of the individual fragments ordered by P1_max
    # from max to min
    index_individual_fragments_sorted_by_P1_max_to_min = np.argsort(P1_max)[::-1]
    return P1_array, index_individual_fragments_sorted_by_P1_max_to_min


def p1_below_random(
    P1_array: np.ndarray, index_individual_fragment: np.ndarray, fragment: Fragment
):
    """Evaluate if a fragment has been assigned with a certainty lower than
    random (wrt the number of possible identities)

    Parameters
    ----------
    P1_array  : nd.array
        P1 vector of a fragment object
    index_individual_fragment  : nd.array
        Argsort of the P1 array of fragment
    fragment : Fragment
        Fragment object containing images associated with a single individual

    Returns
    -------
    p1_below_random_flag : bool
        True if a fragment has been identified with a certainty below random
    """
    return (
        np.max(P1_array[index_individual_fragment, :]) < 1.0 / fragment.number_of_images
    )


def set_fragment_temporary_id(
    fragment: Fragment,
    temporary_id: int,
    P1_array: np.ndarray,
    index_individual_fragment: int,
):
    """Given a P1 array relative to a global fragment sets to 0 the row
    relative to fragment
    which is temporarily identified with identity temporary_id

    Parameters
    ----------
    fragment : Fragment
        Fragment object containing images associated with a single individual
    temporary_id : int
        temporary identifier associated to fragment
    P1_array  : nd.array
        P1 vector of fragment
    index_individual_fragment : int
        Index of fragment with respect to a global fragment in which it is
        contained

    Returns
    -------
    P1_array  : nd.array
        updated P1 array
    """
    fragment.temporary_id = int(temporary_id)
    P1_array[index_individual_fragment, :] = 0.0
    P1_array[:, temporary_id] = 0.0
    return P1_array
