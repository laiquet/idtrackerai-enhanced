from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from scipy.sparse import coo_matrix

from idtrackerai import Fragment
from idtrackerai.tracker.assigner import assign
from idtrackerai.tracker.network.get_predictions import GetPredictionsIdentities

from .network import load_identification_model


def match(model_folder: Path, images: np.ndarray, labels: np.ndarray):
    identification_model, model_params = load_identification_model(model_folder)
    set_of_labels = set(labels)
    if 0 in set_of_labels:
        set_of_labels.remove(0)

    num_labels = len(set_of_labels)
    """number of labels in the images to be assigned by the model (B)"""
    num_classes = model_params.number_of_classes
    """number of classes in the model (A)"""
    # TODO if num_labels <= num_classes:

    confusion_matrix = np.zeros((num_classes, num_labels))
    frequencies_matrix = np.zeros((num_classes, num_labels))
    certainties = np.zeros(num_labels)

    for identity in np.unique(labels).astype(int):
        assigner = assign(
            identification_model, images[labels == identity], model_params
        )
        frequencies, P1_vector, certainty = compute_identification_statistics(assigner)
        confusion_matrix[:, identity - 1] = P1_vector
        frequencies_matrix[:, identity - 1] = frequencies
        certainties[identity - 1] = certainty
    return confusion_matrix, frequencies_matrix, certainties


def get_transfer_dicts(confusion_matrix: np.ndarray, frequencies_matrix: np.ndarray):
    """
    :param confusion_matrix: rows are identities in model, columns are identities in images
    :param frequencies_matrix: rows are identities in model, columns are identities in images
    :return: transfer_dicts: in each dictionary: the key is the identity of the image, the value is the identity assigned by the model
    """
    num_classes = confusion_matrix.shape[0]
    normalized_frequencies = 1.0 - frequencies_matrix / np.tile(
        np.sum(frequencies_matrix, axis=0), [num_classes, 1]
    )
    return {
        "max_P1": get_trasnfer_dict_by_maximum(confusion_matrix),
        "max_freq": get_trasnfer_dict_by_maximum(frequencies_matrix),
        "greedy": get_transfer_dict_greedy(confusion_matrix.copy()),
        "hungarian_P1": get_transfer_dict_by_hungarian(1.0 - confusion_matrix),
        "hungarian_freq": get_transfer_dict_by_hungarian(normalized_frequencies),
    }


def get_transfer_dict_by_hungarian(matrix):
    num_labels = matrix.shape[1]
    r, assign = hungarian(matrix)
    transfer_dict = {
        "assignments_values": {i: None for i in range(1, num_labels + 1)},
        "assignments": {i: None for i in range(1, num_labels + 1)},
    }
    for rr, a in zip(r, assign):
        transfer_dict["assignments"][a + 1] = rr + 1
        transfer_dict["assignments_values"][a + 1] = matrix[rr, a]

    return transfer_dict


def get_trasnfer_dict_by_maximum(matrix):
    num_labels = matrix.shape[1]
    transfer_dict = {
        "assignments_values": {i: None for i in range(1, num_labels + 1)},
        "assignments": {i: None for i in range(1, num_labels + 1)},
    }
    for identity in range(1, num_labels + 1):
        transfer_dict["assignments"][identity] = np.argmax(matrix[:, identity - 1]) + 1
        transfer_dict["assignments_values"][identity] = np.max(matrix[:, identity - 1])

    return transfer_dict


def get_transfer_dict_greedy(confusion_matrix):
    num_labels = confusion_matrix.shape[1]
    transfer_dict = {
        "assignments_values": {i: None for i in range(1, num_labels + 1)},
        "assignments": {i: None for i in range(1, num_labels + 1)},
    }
    max_assigned = np.max(confusion_matrix, axis=0)
    order_of_assignment = np.argsort(max_assigned)[::-1]  # 0 indexed

    for id_to_assign in order_of_assignment:
        id_assigned = np.argmax(confusion_matrix[:, id_to_assign]) + 1
        transfer_dict["assignments"][id_to_assign + 1] = id_assigned
        transfer_dict["assignments_values"][id_to_assign + 1] = np.max(
            confusion_matrix[:, id_to_assign]
        )

        confusion_matrix[id_assigned - 1, :] = 0
        confusion_matrix[:, id_to_assign] = 0

    return transfer_dict


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


def joined_results(matching_results_A_B: dict, matching_results_B_A: dict):
    joined_frequencies_matrix = (
        matching_results_A_B["frequencies_matrix"]
        + matching_results_B_A["frequencies_matrix"].T
    )
    confusion_matrixA = matching_results_A_B[
        "P1_confusion_matrix"
    ]  # rows are model ids cols are images ids
    confusion_matrixB = matching_results_B_A["P1_confusion_matrix"]
    joined_confusion_matrix = 1.0 - (1.0 - confusion_matrixA) * (
        1.0 - confusion_matrixB.T
    )
    joined_transfer_dicts_A_B = get_transfer_dicts(
        joined_confusion_matrix, joined_frequencies_matrix
    )
    joined_transfer_dicts_B_A = get_transfer_dicts(
        joined_confusion_matrix.T, joined_frequencies_matrix.T
    )
    matching_results = {
        "folder_A": matching_results_A_B["network_from"],
        "folder_B": matching_results_B_A["network_from"],
        "joined_frequencies_matrix": joined_frequencies_matrix,
        "joined_P1_confusion_matrix": joined_confusion_matrix,
        "joined_transfer_dict_A_B": joined_transfer_dicts_A_B,
        "joined_transfer_dict_B_A": joined_transfer_dicts_B_A,
        "matches_dict_separated": check_bidirectional_matches(
            matching_results_A_B["transfer_dicts"],
            matching_results_B_A["transfer_dicts"],
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

    return matching_results


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
    transfer_dicts_A_B, transfer_dicts_B_A, num_animals_A, num_animals_B
):
    matches_dict = {}
    for assignment_type in transfer_dicts_A_B.keys():
        matches_dict[assignment_type] = {}
        assignments = transfer_dicts_A_B[assignment_type]["assignments"]
        rows = (
            np.asarray([av for av in assignments.values() if av is not None]) - 1
        )  # identities in model A
        cols = (
            np.asarray([ak for ak in assignments.keys() if assignments[ak] is not None])
            - 1
        )  # identities in images B
        assert len(rows) == len(cols)
        matrix_A_B = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(num_animals_A, num_animals_B)
        ).toarray()

        assignments = transfer_dicts_B_A[assignment_type]["assignments"]
        rows = np.asarray([av for av in assignments.values() if av is not None]) - 1
        cols = (
            np.asarray([ak for ak in assignments.keys() if assignments[ak] is not None])
            - 1
        )
        assert len(rows) == len(cols)
        matrix_B_A = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(num_animals_B, num_animals_A)
        ).toarray()

        matches = np.where(matrix_A_B * matrix_B_A.T == 1)
        matches_dict[assignment_type]["unique_matches"] = list(
            zip(matches[0] + 1, matches[1] + 1)
        )
        missmatches = np.where(matrix_A_B != matrix_B_A.T)
        matches_dict[assignment_type]["missmatches"] = list(
            zip(missmatches[0] + 1, missmatches[1] + 1)
        )

    return matches_dict
