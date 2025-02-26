import logging

import numpy as np

from idtrackerai import ListOfBlobs, conf
from idtrackerai.utils import track


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
        areas = []
        if number_of_animals > 0:
            for blobs_in_frame in list_of_blobs.blobs_in_video:
                if len(blobs_in_frame) == number_of_animals:
                    areas += (blob.area for blob in blobs_in_frame)

        if not areas:
            # either n_animals is 0 or there are no global fragments
            areas = [b.area for b in list_of_blobs.all_blobs]

        areas = np.asarray(areas)
        self.median = np.median(areas)
        self.mean = areas.mean()
        self.std = areas.std()
        self.std_tolerance = conf.MODEL_AREA_SD_TOLERANCE
        self.tolerance = self.std_tolerance * self.std
        logging.info(
            f"Model area computed with {len(areas)} blobs: "
            f"mean={self.mean:.1f}, median={self.median:.1f}, "
            f"and std={self.std:.1f} pixels"
        )

    def __call__(self, area) -> bool:
        return (area - self.median) < self.tolerance


def compute_body_length(list_of_blobs: ListOfBlobs, number_of_animals: int) -> float:
    """computes the median of the the diagonal of the bounding box."""
    # areas are collected throughout the entire video in the cores of
    # the global fragments
    body_lengths = []
    if number_of_animals > 0:
        for blobs_in_frame in track(
            list_of_blobs.blobs_in_video, "Computing body lengths"
        ):
            if len(blobs_in_frame) == number_of_animals:
                body_lengths += (blob.extension for blob in blobs_in_frame)

    if not body_lengths:
        # either n_animals is 0 or there are no global fragments
        body_lengths = [
            b.extension
            for b in track(list_of_blobs.all_blobs, "Computing body lengths")
        ]

    median = np.median(body_lengths)
    logging.info(f"The median body length is {median:.1f} pixels")
    return float(median)
    # return np.percentile(body_lengths, 80)
