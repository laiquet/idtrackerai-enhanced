from collections.abc import Iterator, Sequence
from typing import Literal

from . import Fragment, IdtrackeraiError
from .utils import deprecated


class GlobalFragment:
    """Represents a collection of N_animals coexisting :class:`Fragment` s."""

    duplicated_identities: set

    fragments_identifiers: Sequence[int]

    fragments: Sequence[Fragment]

    accumulation_step: int | None = None
    """Integer indicating the accumulation step at which the global fragment was globally accumulated."""

    def __init__(self, fragments: Sequence[Fragment]) -> None:
        self.fragments_identifiers = tuple(frag.identifier for frag in fragments)
        self.fragments = fragments

        for fragment in self:
            fragment.is_in_a_global_fragment = True

    @property
    def minimum_distance_travelled(self) -> float:
        return min(fragment.distance_travelled for fragment in self)

    @property
    def first_frame_of_the_core(self) -> int:
        return max(frag.start_frame for frag in self)

    @property
    def min_n_images_per_fragment(self) -> int:
        return min(fragment.n_images for fragment in self)

    def __iter__(self) -> Iterator[Fragment]:
        try:
            return iter(self.fragments)
        except AttributeError as exc:
            raise IdtrackeraiError(
                "Global Fragment has been loaded without access to Fragments"
            ) from exc

    @classmethod
    def from_json(
        cls, data: dict, fragments: list[Fragment] | None
    ) -> "GlobalFragment":
        global_fragment = cls.__new__(cls)
        if "individual_fragments_identifiers" in data:
            data["fragments_identifiers"] = data.pop("individual_fragments_identifiers")
        data.pop("first_frame_of_the_core", None)  # it wasn't a property before 6.0.0
        global_fragment.__dict__.update(data)
        if "duplicated_identities" in data:
            global_fragment.duplicated_identities = set(data["duplicated_identities"])

        if fragments is not None:
            global_fragment.fragments = [
                fragments[frag_id] for frag_id in global_fragment.fragments_identifiers
            ]

        return global_fragment

    @property
    def used_for_training(self) -> bool:
        """Boolean indicating if all the fragments in the global fragment
        have been used for training the identification network"""
        return all(fragment.used_for_training for fragment in self)

    def is_unique(self, number_of_animals: int) -> bool:
        """Boolean indicating that the global fragment has unique
        identities, i.e. it does not have duplications."""
        return {fragment.temporary_id for fragment in self} == set(
            range(number_of_animals)
        )

    @property
    def is_partially_unique(self) -> bool:
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

    def acceptable_for_training(
        self, accumulation_strategy: Literal["global", "partial"]
    ) -> bool:
        """Returns True if self is acceptable for training"""

        return (all if accumulation_strategy == "global" else any)(
            fragment.acceptable_for_training for fragment in self
        )

    @property
    def total_number_of_images(self) -> int:
        """Gets the total number of images in self"""
        return sum(fragment.n_images for fragment in self)

    @deprecated(version="6.0.0", action="error")
    def get_images_and_labels(self): ...
