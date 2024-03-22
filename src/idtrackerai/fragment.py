from functools import cached_property
from statistics import fmean
from typing import Any, Iterator, Literal, Sequence

import numpy as np

from .utils import conf


class Fragment:
    """Contains information about a collection of blobs that belong to the
    same animal or to the same crossing."""

    P1_vector: np.ndarray
    """Numpy array indicating the P1 probability of each of the possible
    identities"""

    coexisting_individual_fragments: Sequence["Fragment"]
    """list of fragment objects representing and individual (i.e.
    not representing a crossing where two or more animals are touching) and
    coexisting (in frame) with self. Doesn't include self."""

    frame_by_frame_velocity: np.ndarray
    "Instant speed (in each frame) of the blob in the fragment"

    start_position: tuple[float, float]
    "X and Y position of the blob's centroid at the start of the fragment"

    end_position: tuple[float, float]
    "X and Y position of the blob's centroid at the end of the fragment"

    images: np.ndarray
    "Indices of Fragment's images in the hdf5 files"

    loaded_images: np.ndarray
    "Fragment's actual images, in uint8. Intended to be used outside idtrackerai"

    episodes: np.ndarray
    "Episode where each Fragment images belongs to. It determined the hdf5 file where the image is."

    groundtruth_identity: int
    "Groundtruth identity assigned externally after validating"

    identifier: int

    start_frame: int

    end_frame: int

    is_an_individual: bool

    acceptable_for_training: bool | None = None
    """Boolean to indicate that the fragment was identified sufficiently
    well and can in principle be used for training. See also the
    accumulation_manager.py module."""

    temporary_id: int | None = None
    """Integer indicating a temporary identity assigned to the fragment
    during the cascade of training and identification protocols."""

    accumulable: bool | None = None
    """Boolean indicating whether the fragment can be accumulated, i.e. it
    can potentially be used for training."""

    is_in_a_global_fragment: bool = False
    "Indicates whether the fragment is part of a global fragment"

    certainty: float = 0.0
    "Indicates the certainty of the identity"

    P2_vector: np.ndarray | None = None
    """Numpy array indicating the P2 probability of each of the possible
    identities. See also :meth:`compute_P2_vector`"""

    identity: int | None = None
    """Identity assigned to the fragment during the cascade of training
    and identification protocols or during the residual identification
    (see also the assigner.py module)"""

    non_consistent: bool = False
    """Boolean indicating whether the fragment identity is consistent with
    coexisting fragment"""

    ambiguous_identities: np.ndarray | None = None
    """Identities that would be ambiguously assigned during the residual
    identification process. See also the assigner.py module"""

    used_for_training: bool = False
    """Boolean indicating whether the images in the fragment were used to
    train the identification network during the cascade of training and
    identification protocols. See also the accumulation_manager.py module.
    """
    accumulation_step: int | None = None
    """Integer indicating the accumulation step at which the fragment was
    accumulated. See also the accumulation_manager.py module."""

    identities_corrected_closing_gaps: list[int] | None = None
    """Identity of the fragment assigned during the interpolation of the
        gaps produced by the crossing fragments. See also the
        assign_them_all.py module."""

    identity_corrected_solving_jumps: int | None = None
    """Identity of the fragment assigned during the correction of imposible
    (unrealistic) velocity jumps in the trajectories. See also the
    correct_impossible_velocity_jumps.py module."""

    identity_is_fixed: bool = False
    """Boolean indicating whether the identity is fixed and cannot be
    modified during the postprocessing. This attribute is given during
    the residual identification (see assigner.py module)"""

    P1_below_random: bool = False

    used_for_pretraining: bool = False
    """Boolean indicating whether the images in the fragment were used to
    pretrain the identification network during the pretraining step of the
    Protocol 3. See also the accumulation_manager.py module."""

    accumulated_globally: bool = False
    """Boolean indicating whether the fragment was accumulated in a
    global accumulation step of the cascade of training and identification
    protocols. See also the accumulation_manager.py module."""

    accumulated_partially: bool = False
    """Boolean indicating whether the fragment was accumulated in a
    partial accumulation step of the cascade of training and identification
    protocols. See also the accumulation_manager.py module."""

    forced_crossing: bool = False
    "Indicates if the crossing attribute has been forced by set_individual_with_identity_0_as_crossings()"

    exclusive_roi: int = -1
    "Exclusive ROI where the fragment belongs to. -1 for disabled exclusive ROIs"

    zero_identity_assigned_by_P2: bool = False
    zero_identity_assigned_by_exclusive_rois: bool = False

    def __init__(
        self,
        fragment_identifier: int,
        start_frame: int,
        end_frame: int,
        images: list[int] | np.ndarray,
        centroids: list[tuple[float, float]],
        episodes: list[int] | np.ndarray,
        is_an_individual: bool,
        exclusive_roi: int,
    ) -> None:
        self.identifier = fragment_identifier
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.images = np.asarray(images)
        self.episodes = np.asarray(episodes)
        self.is_an_individual = is_an_individual
        self.exclusive_roi = exclusive_roi

        if len(centroids) > 1:
            self.frame_by_frame_velocity = np.sqrt(
                (np.diff(np.asarray(centroids), axis=0) ** 2).sum(axis=1)
            )
        else:
            self.frame_by_frame_velocity = np.array([0])

        self.start_position = centroids[0]
        self.end_position = centroids[-1]

    def __len__(self) -> int:
        return self.n_images

    @property
    def image_locations(self) -> Iterator[tuple[Any, Any]]:
        return zip(self.images, self.episodes)

    @classmethod
    def from_json(cls, json: dict) -> "Fragment":
        fragment: cls = cls.__new__(cls)
        fragment.__dict__ = json
        if len(fragment.episodes) == 1:  # v<=5.2.5 decompress
            fragment.episodes = np.full(len(fragment), fragment.episodes[0])
        if (
            len(fragment.episodes) == 2
            and isinstance(fragment.episodes[0], list)
            and isinstance(fragment.episodes[1], list)
        ):  # v>5.2.5 decompress
            fragment.episodes = np.repeat(*fragment.episodes)
        else:
            fragment.episodes = np.asarray(fragment.episodes)

        for key in (
            "images",
            "P1_vector",
            "P2_vector",
            "ambiguous_identities",
            "frame_by_frame_velocity",
        ):
            if key in json:
                setattr(fragment, key, np.asarray(json[key]))
        return fragment

    @property
    def distance_travelled(self) -> float:
        """The distance traveled by the individual in the fragment.
        It is based on the position of the centroids in consecutive images.
        """
        return self.frame_by_frame_velocity.sum()

    def reset(
        self,
        roll_back_to: Literal["fragmentation", "accumulation"],
        number_of_animals: int,
    ) -> None:
        """Reset attributes of the fragment to a specific part of the
        algorithm.

        Parameters
        ----------
        roll_back_to : str
            Reset all the attributes up to the process specified in input.
            'fragmentation', 'pretraining', 'accumulation', 'assignment'
        """
        #  This method was mainly used to resume the tracking from different
        # rocessing steps. Currently this function is not active, but this
        #  method might still be useful in the future.
        self.identity_is_fixed = False
        if roll_back_to == "fragmentation":
            self.used_for_training = False
            self.used_for_pretraining = False
            self.acceptable_for_training = None
            self.temporary_id = None
            self.identity = None
            self.identity_corrected_solving_jumps = None
            self.accumulated_globally = False
            self.accumulated_partially = False
            self.accumulation_step = None
            self.non_consistent = False
            self.certainty = 0.0
            self.P1_vector = np.zeros(number_of_animals)
            self.P1_below_random = False
        elif roll_back_to == "accumulation":
            if not self.used_for_training:
                self.identity = None
                self.identity_corrected_solving_jumps = None
                self.P1_vector = np.zeros(number_of_animals)
            self.ambiguous_identities = None
            self.P2_vector = None
            self.__dict__.pop("certainty_P2", None)
        else:
            raise ValueError(roll_back_to)

    @property
    def is_a_crossing(self) -> bool:
        return not self.is_an_individual

    @property
    def assigned_identities(self) -> list[int] | list[int | None]:
        """Assigned identities (list) by the algorithm considering the
        identification process and the postprocessing steps (correction of
        impossible velocity jumps and interpolation of crossings).

        The fragment can have multiple identities if it is a crossing fragment.
        """
        if self.identities_corrected_closing_gaps is not None:
            return self.identities_corrected_closing_gaps
        if self.identity_corrected_solving_jumps is not None:
            return [self.identity_corrected_solving_jumps]
        return [self.identity]

    @cached_property
    def n_images(self) -> int:
        """Number images (or blobs) in the fragment."""
        return len(self.images)

    @property
    def is_certain(self) -> bool:
        """Whether the fragment is certain enough to be accumulated."""
        return self.certainty >= conf.CERTAINTY_THRESHOLD

    @property
    def has_enough_accumulated_coexisting_fragments(self) -> bool:
        """Whether the fragment has enough coexisting and
        already accumulated fragments (the threshold is half of them).

        This property is used during the partial accumulation. See also the
        accumulation_manager.py module.
        """
        return (
            fmean(
                fragment.used_for_training
                for fragment in self.coexisting_individual_fragments
            )
            >= 0.5
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("coexisting_individual_fragments", None)
        state.pop("centroids", None)  # v5.1.3 compatibility
        state.pop("accumulable", None)
        state.pop("n_images", None)  # cached_property
        state.pop("certainty_P2", None)  # cached_property
        return state

    def compute_border_velocity(self, other: "Fragment|None") -> float | None:
        """Velocity necessary to cover the space between two fragments.

        Note that these velocities are divided by the number of frames that
        separate self and other fragment.

        Parameters
        ----------
        other : :class:`Fragment`
            Another fragment

        Returns
        -------
        float
            Returns the speed at which an individual should travel to be
            present in both self and other fragments.

        """
        if other is None:
            return None
        if self.start_frame > other.end_frame:
            centroids = np.asarray([self.start_position, other.end_position])
        else:
            centroids = np.asarray([self.end_position, other.start_position])
        return np.sqrt((np.diff(centroids, axis=0) ** 2).sum(axis=1))[0]

    def coexist_with(self, other: "Fragment") -> bool:
        """Boolean indicating whether the given fragment coexists in time with
        another fragment.

        Parameters
        ----------
        other :  :class:`Fragment`
            A second fragment

        Returns
        -------
        bool
            True if self and other coexist in time in at least one frame.

        """
        return self.start_frame < other.end_frame and self.end_frame > other.start_frame

    def is_inconsistent_with_coexistent_fragments(self, temporary_id) -> bool:
        """Check that the temporary identity assigned to the fragment is
        consistent with respect to the identities already assigned to the
        fragments coexisting (in frame) with it.

        Parameters
        ----------
        temporary_id : int
            Temporary identity assigned to the fragment.

        Returns
        -------
        bool
            True if the identification of self with `temporary_id` does not
            cause any duplication of identities.

        """
        return any(
            coexisting_fragment.temporary_id == temporary_id
            for coexisting_fragment in self.coexisting_individual_fragments
        )

    def compute_identification_statistics(
        self,
        predictions: np.ndarray,
        softmax_probs: np.ndarray | None,
        number_of_animals: int,
    ) -> None:
        """Computes the statistics necessary for the identification of the
        fragment.

        Parameters
        ----------
        predictions : numpy array
            Array of shape [number_of_images_in_fragment, 1] whose components
            are the argmax(softmax_probs) per image
        softmax_probs : numpy array
            Array of shape [number_of_images_in_fragment, 1]
            whose values is the maximum result of applying the softmax function to the
            predictions outputted by the idCNN per image
        number_of_animals : int
            Description of parameter `number_of_animals`.

        See Also
        --------
        :meth:`compute_median_softmax`
        """
        assert self.is_an_individual
        assert len(predictions) == self.n_images

        frequencies = np.bincount(predictions, minlength=number_of_animals + 1)[1:]
        self.set_P1_from_frequencies(frequencies)
        if softmax_probs is not None:
            median_softmax = self.compute_median_softmax(
                softmax_probs, predictions, number_of_animals
            )
            self.set_certainty_of_individual_fragment(median_softmax)

    def assign_identity(
        self, number_of_animals: int, id_to_roi: list[int] | np.ndarray
    ) -> None:
        """Assigns the identity to the fragment by considering the fragments
        coexisting with it.

        If the certainty of the identification is high enough it sets
        the identity of the fragment as fixed and it won't be modified during
        the postprocessing.
        """
        assert self.is_an_individual
        assert not self.used_for_training
        assert not self.identity_is_fixed
        assert self.identity is None
        assert self.P2_vector is not None

        max_P2 = self.P2_vector.max()  # there can be two equal maximums
        possible_identities = np.nonzero(self.P2_vector == max_P2)[0] + 1

        if len(possible_identities) > 1:
            self.identity = 0
            self.zero_identity_assigned_by_P2 = True
            self.ambiguous_identities = possible_identities
            return

        identity = possible_identities[0]
        if id_to_roi[identity - 1] != self.exclusive_roi:
            self.identity = 0
            self.zero_identity_assigned_by_exclusive_rois = True
            return

        self.identity = identity
        if (
            max_P2 > conf.FIXED_IDENTITY_THRESHOLD
            and self.n_images
            > conf.MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
        ):
            self.identity_is_fixed = True
        self.P1_vector = np.zeros(len(self.P1_vector))
        self.P1_vector[self.identity - 1] = 1.0
        for fragment in self.coexisting_individual_fragments:
            fragment.compute_P2_vector(number_of_animals, only_non_identified=True)

    def compute_P2_vector(
        self, number_of_animals: int, only_non_identified=False
    ) -> None:
        """Computes the P2_vector of the fragment.

        The flag only_non_identified is to save computational resources when
        Assigning identities after accumulation"""
        self.__dict__.pop("certainty_P2", None)  # clear cached property
        if only_non_identified and self.identity is not None:
            return
        coexisting_P1_vectors = np.asarray(
            [fragment.P1_vector for fragment in self.coexisting_individual_fragments]
        )
        numerator = self.P1_vector * np.prod(1.0 - coexisting_P1_vectors, axis=0)
        denominator = numerator.sum()
        if denominator != 0:
            self.P2_vector = numerator / denominator
        else:
            self.P2_vector = np.zeros(number_of_animals)

    @cached_property
    def certainty_P2(self) -> float:
        """Indicating the certainty of the identity following the P2"""

        if self.P2_vector is None or self.P2_vector.sum() < 0.001:
            return 0.0

        second_max, first_max = np.sort(self.P2_vector)[-2:]

        with np.errstate(divide="ignore", over="ignore"):
            return first_max / second_max

    def set_P1_from_frequencies(self, frequencies: np.ndarray) -> None:
        """Given the frequencies of a individual fragment
        computer the P1 vector.

        P1 is the softmax of the frequencies with base 2 for each identity.
        Numpy array indicating the number of images assigned with each of
        the possible identities
        """
        with np.errstate(over="ignore"):
            self.P1_vector = 1.0 / (
                2.0
                ** (
                    np.tile(frequencies, (len(frequencies), 1)).T
                    - np.tile(frequencies, (len(frequencies), 1))
                )
            ).sum(axis=0)

    @staticmethod
    def compute_median_softmax(
        softmax_probs: np.ndarray, preditions: np.ndarray, number_of_animals
    ) -> np.ndarray:
        """Given the softmax of the predictions outputted by the identification
        network, it computes their median according to the argmax of the
        softmaxed predictions per image.

        Parameters
        ----------
        softmax_probs : ndarray
            array of shape [number_of_images_in_fragment, number_of_animals]
            whose rows are the result of applying the softmax function to the
            predictions outputted by the idCNN per image
        number_of_animals : int
            number of animals to be tracked as defined by the user

        Returns
        -------
        float
            Median of argmax(softmax_probs) per identity

        """
        assert softmax_probs.ndim == 1
        softmax_median = np.zeros(number_of_animals)
        for i in np.unique(preditions):
            softmax_median[i - 1] = np.median(softmax_probs[preditions == i])
        return softmax_median

    def set_certainty_of_individual_fragment(self, median_softmax: np.ndarray) -> None:
        """Computes the certainty given the P1_vector of the fragment by
        using the output of :meth:`compute_median_softmax`

        Parameters
        ----------
        P1_vector : numpy array
            Array with shape [1, number_of_animals] computed from frequencies
            by :meth:`compute_identification_statistics`
        median_softmax : ndarray
            Median of argmax(softmax_probs) per image

        Returns
        -------
        float
            Fragment's certainty

        """
        argsort_p1_vector = self.P1_vector.argsort()
        sorted_p1_vector = self.P1_vector[argsort_p1_vector]
        sorted_softmax_probs = median_softmax[argsort_p1_vector]
        self.certainty = (
            np.diff((sorted_p1_vector * sorted_softmax_probs)[-2:])
            / sorted_p1_vector[-2:].sum()
        )[0]

    def get_neighbour_fragment(
        self,
        fragments: Sequence["Fragment"],
        scope: Literal["to_the_past", "to_the_future"],
    ) -> "Fragment | None":
        """If it exist, gets the fragment in the list of all fragment whose
        identity is the identity assigned to self and whose starting frame is
        the ending frame of self + 1, or ending frame is the starting frame of
        self - 1

        Parameters
        ----------
        fragments : list
            List of all the fragments in the video
        scope : str
            If "to_the_future" looks for the consecutive fragment wrt to self,
            if "to_the_past" looks for the fragment the precedes self
        number_of_frames_in_direction : int
            Distance (in frame) at which the previous or next fragment has to
            be

        Returns
        -------
        :class:`fragment.Fragment`
            The neighbouring fragment with respect to self in the direction
            specified by scope if it exists. Otherwise None

        """
        self_index = fragments.index(self)
        if scope == "to_the_past":
            for frag in fragments[self_index - 1 :: -1]:
                if (
                    frag.is_an_individual
                    and frag.assigned_identities[0] == self.assigned_identities[0]
                    and self.start_frame == frag.end_frame
                ):
                    assert len(frag.assigned_identities) == 1
                    return frag

        elif scope == "to_the_future":
            for frag in fragments[self_index + 1 :]:
                if (
                    frag.is_an_individual
                    and frag.assigned_identities[0] == self.assigned_identities[0]
                    and frag.start_frame == self.end_frame
                ):
                    assert len(frag.assigned_identities) == 1
                    return frag
                if frag.start_frame > self.end_frame:
                    # next fragments with have larger and larger start_frame, no chances to find it
                    break

        else:
            raise ValueError(scope)

        return None

    def set_partially_or_globally_accumulated(self, accumulation_strategy) -> None:
        """Sets :attr:`accumulated_globally` and :attr:`accumulated_partially`
        according to `accumulation_strategy`.

        Parameters
        ----------
        accumulation_strategy : str
            Can be "global" or "partial"

        """
        if accumulation_strategy == "global":
            self.accumulated_globally = True
        elif accumulation_strategy == "partial":
            self.accumulated_partially = True

    @property
    def properties(self) -> Sequence[str]:
        return (
            f"Fragment {self.identifier}",
            (
                f"Frames from {self.start_frame} to {self.end_frame} (length"
                f" {self.end_frame-self.start_frame})"
            ),
            ("Individual" if self.is_an_individual else "Crossing")
            + " fragment"
            + (" (forced)" if self.forced_crossing else ""),
            ("Used" if self.used_for_training else "Not used") + " for training",
            ("Used" if self.used_for_pretraining else "Not used") + " for pretraining",
            ("Acceptable" if self.acceptable_for_training else "Not acceptable")
            + " for training",
            f"Predicted identity: {self.identity}",
            f"Corrected solving jumps: {self.identity_corrected_solving_jumps}",
            f"Corrected solving gaps: {self.identities_corrected_closing_gaps}",
            f"Fixed identity: {self.identity_is_fixed}",
            f"Globally accumulated: {self.accumulated_globally}",
            f"Partially accumulated: {self.accumulated_partially}",
            f"Accumulable: {self.accumulable}",
            f"Accumulated at step {self.accumulation_step}",
            "Non consistent" if self.non_consistent else "Consistent",
            (
                f"Max P1 {np.argmax(self.P1_vector)+1} with value"
                f" {self.P1_vector.max()}"
                if hasattr(self, "P1_vector")
                else "Doesn't have P1 vector"
            ),
            f"Certainty: {self.certainty}",
            f"P1 below random: {self.P1_below_random}",
        )
