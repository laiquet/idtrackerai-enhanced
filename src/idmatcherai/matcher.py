import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from idtrackerai import Fragment
from idtrackerai.tracker.network.get_predictions import GetPredictionsIdentities

from .images import extact_all_images_and_labels
from .network import load_identification_model


def match(id_images_path: Path, model_path: Path):
    logging.info(
        "Matching images from %s with model from %s", id_images_path, model_path
    )

    images, labels = extact_all_images_and_labels(
        id_images_path.glob("id_images_*.hdf5")
    )
    model, model_params = load_identification_model(model_path)

    set_of_labels = set(labels.astype(int))
    set_of_labels.discard(0)

    num_labels = len(set_of_labels)
    """number of labels in the images to be assigned by the model (B)"""
    num_classes = model_params.number_of_classes
    """number of classes in the model (A)"""
    # TODO if num_labels <= num_classes:

    confusion_matrix = np.zeros((num_classes, num_labels))
    frequencies_matrix = np.zeros((num_classes, num_labels), int)
    certainties = np.zeros(num_labels)

    for identity in set_of_labels:
        assigner = GetPredictionsIdentities(
            model, images[labels == identity], model_params
        )
        assigner.get_all_predictions()
        frequencies, P1_vector, certainty = compute_identification_statistics(assigner)
        confusion_matrix[identity - 1] = P1_vector
        frequencies_matrix[identity - 1] = frequencies
        certainties[identity - 1] = certainty
    return confusion_matrix, frequencies_matrix, certainties


def get_transfer_dicts(confusion_matrix: np.ndarray, frequencies_matrix: np.ndarray):
    """
    :param confusion_matrix: rows are identities in model, columns are identities in images
    :param frequencies_matrix: rows are identities in model, columns are identities in images
    :return: transfer_dicts: in each dictionary: the key is the identity of the image, the value is the identity assigned by the model
    """
    return {
        "hungarian_P1": get_transfer_dict_by_hungarian(confusion_matrix),
        "hungarian_freq": get_transfer_dict_by_hungarian(
            frequencies_matrix / np.sum(frequencies_matrix, axis=1)[:, None]
        ),
    }


def get_transfer_dict_by_hungarian(matrix):
    r, assign = linear_sum_assignment(matrix, maximize=True)
    return {"assignments_values": matrix[r, assign], "assignments": r + 1}


def compute_identification_statistics(assigner: GetPredictionsIdentities):
    frequencies = Fragment.compute_identification_frequencies_individual_fragment(
        assigner._predictions, assigner.network_params.number_of_classes
    )
    P1_vector = Fragment.compute_P1_from_frequencies(frequencies)
    median_softmax = Fragment.compute_median_softmax(
        assigner._softmax_probs, assigner.network_params.number_of_classes
    )
    certainty = Fragment.compute_certainty_of_individual_fragment(
        P1_vector, median_softmax
    )
    return frequencies, P1_vector, certainty


def joined_results(match_AB: dict, match_BA: dict):
    joined_frequencies_matrix = (
        match_AB["frequencies_matrix"] + match_BA["frequencies_matrix"].T
    )
    confusion_matrixA = match_AB[
        "P1_confusion_matrix"
    ]  # rows are model ids cols are images ids
    confusion_matrixB = match_BA["P1_confusion_matrix"]
    joined_confusion_matrix = 1.0 - (1.0 - confusion_matrixA) * (
        1.0 - confusion_matrixB.T
    )
    joined_transfer_dicts_A_B = get_transfer_dicts(
        joined_confusion_matrix, joined_frequencies_matrix
    )
    joined_transfer_dicts_B_A = get_transfer_dicts(
        joined_confusion_matrix.T, joined_frequencies_matrix.T
    )
    return {
        "folder_A": match_AB["network_from"],
        "folder_B": match_BA["network_from"],
        "joined_frequencies_matrix": joined_frequencies_matrix,
        "joined_P1_confusion_matrix": joined_confusion_matrix,
        "joined_transfer_dict_A_B": joined_transfer_dicts_A_B,
        "joined_transfer_dict_B_A": joined_transfer_dicts_B_A,
        "matches_dict_separated": check_bidirectional_matches(
            match_AB["transfer_dicts"],
            match_BA["transfer_dicts"],
            confusion_matrixA.shape[0],
            confusion_matrixA.shape[1],
        ),
        "matches_dict_joined": check_bidirectional_matches(
            joined_transfer_dicts_A_B,
            joined_transfer_dicts_B_A,
            confusion_matrixA.shape[0],
            confusion_matrixA.shape[1],
        ),
    }


def check_missmatches_confusion_matrix(confusion_matrix, threshold=0.99):
    rows, cols = np.where(confusion_matrix > threshold)
    rows, cols = list(rows + 1), list(cols + 1)
    missmatched_identities = {x for x in rows if rows.count(x) > 1}
    missmatches = {
        id: list(np.where(confusion_matrix[id - 1, :] > threshold)[0] + 1)
        for id in missmatched_identities
    }
    return missmatches


def compute_unique_matches_by_confusion_matrix(confusion_matrix, threshold=0.99):
    rows, cols = np.where(confusion_matrix > threshold)
    rows, cols = list(rows + 1), list(cols + 1)
    matches = zip(rows, cols)
    unique_matches = [
        m for m in matches if rows.count(m[0]) == 1 and cols.count(m[1]) == 1
    ]
    return unique_matches


def check_bidirectional_matches(
    transfer_dicts_A_B: dict,
    transfer_dicts_B_A: dict,
    n_animals_A: int,
    n_animals_B: int,
):
    matches_dict = {}
    for assignment_type in transfer_dicts_A_B.keys():
        matrix_A_B = np.zeros((n_animals_A, n_animals_B), bool)
        assignments = transfer_dicts_A_B[assignment_type]["assignments"]
        matrix_A_B[range(len(assignments)), assignments - 1] = True

        matrix_B_A = np.zeros((n_animals_B, n_animals_A), bool)
        assignments = transfer_dicts_B_A[assignment_type]["assignments"]
        matrix_B_A[range(len(assignments)), assignments - 1] = True

        matches = np.where(matrix_A_B * matrix_B_A.T)
        missmatches = np.where(np.logical_xor(matrix_A_B, matrix_B_A.T))
        matches_dict[assignment_type] = {
            "unique_matches": list(zip(matches[0] + 1, matches[1] + 1)),
            "missmatches": list(zip(missmatches[0] + 1, missmatches[1] + 1)),
        }

    return matches_dict
