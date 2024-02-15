from typing import Literal, Sequence

import numpy as np

from . import Blob, Fragment


class GlobalFragment:
    """Represents a collection of :class:`fragment.Fragment` N different
    animals. Where N is the number of animals in the video as defined by the
    user.
    """

    duplicated_identities: set

    first_frame_of_the_core: int

    fragments_identifiers: Sequence[int]

    fragments: list[Fragment]

    minimum_distance_travelled: float

    accumulation_step: int | None = None
    """Integer indicating the accumulation step at which the global fragment
    was globally accumulated."""

    def __init__(
        self,
        blobs_in_video: list[list[Blob]],
        fragments: list[Fragment],
        first_frame_of_the_core: int,
    ):
        self.first_frame_of_the_core = first_frame_of_the_core
        self.fragments_identifiers = tuple(
            blob.fragment_identifier for blob in blobs_in_video[first_frame_of_the_core]
        )
        self.set_individual_fragments(fragments)

        for fragment in self:
            fragment.is_in_a_global_fragment = True

        self.minimum_distance_travelled = min(
            fragment.distance_travelled for fragment in self
        )

    @property
    def min_n_images_per_fragment(self):
        return min(fragment.n_images for fragment in self)

    def __iter__(self):
        return iter(self.fragments)

    @classmethod
    def from_json(cls, data: dict, fragments: list[Fragment] | None):
        global_fragment = cls.__new__(cls)
        if "individual_fragments_identifiers" in data:
            data["fragments_identifiers"] = data.pop("individual_fragments_identifiers")
        global_fragment.__dict__.update(data)
        if "duplicated_identities" in data:
            global_fragment.duplicated_identities = set(data["duplicated_identities"])

        if fragments is not None:
            global_fragment.set_individual_fragments(fragments)

        return global_fragment

    @property
    def used_for_training(self):
        """Boolean indicating if all the fragments in the global fragment
        have been used for training the identification network"""
        return all(fragment.used_for_training for fragment in self)

    def is_unique(self, number_of_animals: int):
        """Boolean indicating that the global fragment has unique
        identities, i.e. it does not have duplications."""
        return {fragment.temporary_id for fragment in self} == set(
            range(number_of_animals)
        )

    @property
    def is_partially_unique(self):
        """Boolean indicating that a subset of the fragments in the global
        fragment have unique identities"""

        identities_acceptable_for_training = [
            fragment.temporary_id
            for fragment in self
            if fragment.acceptable_for_training
        ]
        self.duplicated_identities = {
            x
            for x in identities_acceptable_for_training
            if identities_acceptable_for_training.count(x) > 1
        }
        return len(self.duplicated_identities) == 0

    def set_individual_fragments(self, fragments: Sequence[Fragment]):
        """Gets the list of instances of the class :class:`fragment.Fragment`
        that constitute the global fragment and sets an attribute with such
        list.

        Parameters
        ----------
        fragments : list
            All the fragments extracted from the video.

        """
        self.fragments = [
            fragments[identifier] for identifier in self.fragments_identifiers
        ]

    def acceptable_for_training(
        self, accumulation_strategy: Literal["global", "partial"]
    ) -> bool:
        """Returns True if the global fragment is acceptable for training"""

        return (all if accumulation_strategy == "global" else any)(
            fragment.acceptable_for_training for fragment in self
        )

    @property
    def total_number_of_images(self) -> int:
        """Gets the total number of images in the global fragment"""
        return sum(fragment.n_images for fragment in self)

    def get_images_and_labels(self):
        """Gets the images and identities in the global fragment as a
        labelled dataset in order to train the identification neural network

        If the scope is "pretraining" the identities of each fragment
        will be arbitrary.
        If the scope is "identity_transfer" then the labels will be
        empty as they will be infered by the identification network selected
        by the user to perform the transferring of identities.

        Parameters
        ----------
        id_images_file_paths : list
            List of paths (str) where the identification images are stored.
        scope : str, optional
            Whether the images are going to be used for training the
            identification network or for "pretraining", by default
            "pretraining".

        Returns
        -------
        Tuple
            Tuple with two Numpy arrays with the images and their labels.
        """
        images: list[tuple[int, int]] = []
        labels: list[int] = []

        for temporary_id, fragment in enumerate(self):
            images += fragment.image_locations
            labels += [temporary_id] * fragment.n_images

        return np.asarray(images), np.asarray(labels)
