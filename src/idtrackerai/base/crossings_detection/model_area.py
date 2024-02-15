import logging

import numpy as np

from idtrackerai import ListOfBlobs
from idtrackerai.utils import IdtrackeraiError, conf, track


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
            raise IdtrackeraiError(
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
        for blobs_in_frame in track(
            list_of_blobs.blobs_in_video, "Computing body lengths"
        ):
            if len(blobs_in_frame) == number_of_animals:
                for blob in blobs_in_frame:
                    body_lengths.append(blob.estimated_body_length)
    else:
        body_lengths = [
            b.estimated_body_length
            for b in track(list_of_blobs.all_blobs, "Computing body lengths")
        ]

    median = np.median(body_lengths)
    logging.info(f"Median body length: {median} pixels")
    return float(median)
    # return np.percentile(body_lengths, 80)
