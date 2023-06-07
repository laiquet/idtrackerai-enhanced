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
from typing import Literal, Sequence

import numpy as np

from .utils import conf


class Fragment:
    """Contains information about a collection of blobs that belong to the
    same animal or to the same crossing.

    Parameters
    ----------
    fragment_identifier : int
        It uniquely identifies the fragment.
        It is also used to link blobs to fragments, as blobs have an attribute
        called `blob.Blob.fragment_identifier`.
    start_end : tuple
        Indicates the start and end of the fragment.
        The end is exclusive, i.e. follows Python standards.
    images : list
        List of integers indicating the index of the identification image
        in the episode.
        This corresponds to the `identification_image_index` of the blob.
        Note that the images are stored in the identification_images folder
        inside of the session folder.
        Then the images are loaded using this index and the episode index.
    centroids : list
        List of tuples (x, y) with the centroid of each blob in the fragment.
        The centroids are in pixels and consider the resolution_reduction
        factor.
    episodes : list
        List of integers indicating the episode corresponding to the
        equivalent image index.
    is_an_individual : bool
        Indicates whether the fragment corresponds to a collection of blobs
        that are all labelled as being an individual.
    is_a_crossing : bool
        Indicates whether the fragment corresponds to a collection of blobs
        that are all labelled as being a crossing.
    """

    acceptable_for_training: bool | None
    """Boolean to indicate that the fragment was identified sufficiently
    well and can in principle be used for training. See also the
    accumulation_manager.py module."""

    temporary_id: int | None
    """Integer indicating a temporary identity assigned to the fragment
    during the cascade of training and identification protocols."""

    is_certain: bool | None = None
    """Boolean indicating whether the fragment is certain enough to be
    accumulated. See also the accumulation_manager.py module."""

    accumulable: bool | None = None
    """Boolean indicating whether the fragment can be accumulated, i.e. it
    can potentially be used for training."""

    is_in_a_global_fragment: bool = False
    """Indicates whether the fragment is part of a global fragment"""

    P1_vector: np.ndarray
    """Numpy array indicating the P1 probability of each of the possible
    identities"""

    certainty: float
    """Indicates the certainty of the identity"""

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

    P1_below_random: bool | None

    used_for_pretraining = False
    """Boolean indicating whether the images in the fragment were used to
    pretrain the identification network during the pretraining step of the
    Protocol 3. See also the accumulation_manager.py module."""

    accumulated_globally: bool = False
    """Boolean indicating whether the fragment was accumulated in a
    global accumulation step of the cascade of training and identification
    protocols. See also the accumulation_manager.py module."""

    accumulated_partially = False
    """Boolean indicating whether the fragment was accumulated in a
    partial accumulation step of the cascade of training and identification
    protocols. See also the accumulation_manager.py module."""

    user_generated_identity: int | None = None
    """This property is give during the correction of impossible velocity
    jumps. It has nothing to do with the manual validation."""

    coexisting_individual_fragments: list["Fragment"]
    """list of fragment objects representing and individual (i.e.
    not representing a crossing where two or more animals are touching) and
    coexisting (in frame) with self"""

    forced_crossing: bool = False
    """Indicates if the crossing attribute has been forced by set_individual_with_identity_0_as_crossings()"""

    frame_by_frame_velocity: np.ndarray
    """Instant speed (in each frame) of the blob in the fragment"""

    start_position: tuple[float, float]
    """X and Y position of the blob's centroid at the start of the fragment"""

    end_position: tuple[float, float]
    """X and Y position of the blob's centroid at the end of the fragment"""

    def __init__(
        self,
        fragment_identifier: int,
        start_frame: int,
        end_frame: int,
        images: list[int],
        centroids: list[tuple[float, float]],
        episodes: list[int],
        is_an_individual: bool,
    ):
        self.identifier = fragment_identifier
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.images = images
        self.episodes = episodes
        self.is_an_individual = is_an_individual

        if len(centroids) > 1:
            self.frame_by_frame_velocity = np.sqrt(
                (np.diff(centroids, axis=0) ** 2).sum(axis=1)
            )
        else:
            self.frame_by_frame_velocity = np.array([0])

        self.start_position = centroids[0]
        self.end_position = centroids[-1]

    @property
    def image_locations(self):
        return zip(self.images, self.episodes)

    @classmethod
    def from_json(cls, json: dict):
        fragment: cls = cls.__new__(cls)
        fragment.__dict__ = json
        if len(fragment.episodes) == 1:  # decompress
            fragment.episodes = [fragment.episodes[0]] * len(fragment.images)
        for key in (
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
        return np.sum(self.frame_by_frame_velocity)

    def reset(
        self,
        roll_back_to: Literal["fragmentation", "accumulation"],
        number_of_animals: int,
    ):
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
            self.is_certain = None
            self.non_consistent = False
            self.certainty = 0.0
            self.P1_vector = np.zeros(number_of_animals)
            self.P1_below_random = None
        elif roll_back_to == "accumulation":
            if not self.used_for_training:
                self.identity = None
                self.identity_corrected_solving_jumps = None
                self.P1_vector = np.zeros(number_of_animals)
            self.ambiguous_identities = None
            self.P2_vector = None
        else:
            raise ValueError(roll_back_to)

    @property
    def is_a_crossing(self) -> bool:
        return not self.is_an_individual

    @property
    def assigned_identities(self):
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

    @property
    def number_of_images(self):
        """Number images (or blobs) in the fragment."""
        return len(self.images)

    @property
    def has_enough_accumulated_coexisting_fragments(self):
        """Boolean indicating whether the fragment has enough coexisting and
        already accumulated fragments.

        This property is used during the partial accumulation. See also the
        accumulation_manager.py module.
        """
        return (
            sum(
                fragment.used_for_training
                for fragment in self.coexisting_individual_fragments
            )
            >= self.number_of_coexisting_individual_fragments / 2
        )

    def compute_border_velocity(self, other: "Fragment") -> float:
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
        if self.start_frame > other.end_frame:
            centroids = np.asarray([self.start_position, other.end_position])
        else:
            centroids = np.asarray([self.end_position, other.start_position])
        return np.sqrt((np.diff(centroids, axis=0) ** 2).sum(axis=1))[0]

    def coexist_with(self, other: "Fragment"):
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

    @property
    def number_of_coexisting_individual_fragments(self):
        return len(self.coexisting_individual_fragments)

    def check_consistency_with_coexistent_individual_fragments(self, temporary_id):
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
        return all(
            coexisting_fragment.temporary_id != temporary_id
            for coexisting_fragment in self.coexisting_individual_fragments
        )

    def compute_identification_statistics(
        self, predictions: np.ndarray | list, softmax_probs, number_of_animals: int
    ):
        """Computes the statistics necessary for the identification of the
        fragment.

        Parameters
        ----------
        predictions : numpy array
            Array of shape [number_of_images_in_fragment, 1] whose components
            are the argmax(softmax_probs) per image
        softmax_probs : numpy array
            Array of shape [number_of_images_in_fragment, number_of_animals]
            whose rows are the result of applying the softmax function to the
            predictions outputted by the idCNN per image
        number_of_animals : int
            Description of parameter `number_of_animals`.

        See Also
        --------
        :meth:`compute_median_softmax`
        :meth:`compute_certainty_of_individual_fragment`
        """
        assert self.is_an_individual

        self.P1_vector = self.compute_P1_from_frequencies(
            np.bincount(predictions, minlength=number_of_animals + 1)[1:]
        )
        median_softmax = self.compute_median_softmax(softmax_probs, number_of_animals)
        self.certainty = self.compute_certainty_of_individual_fragment(
            self.P1_vector, median_softmax
        )

    def set_P1_vector_accumulated(self):
        """If the fragment has been used for training its P1_vector is
        modified to be a vector of zeros with a single component set to 1 in
        the :attr:`temporary_id` position.
        """
        assert self.used_for_training and self.is_an_individual
        self.P1_vector[:] = 0.0
        self.P1_vector[self.temporary_id] = 1.0

    @staticmethod
    def get_possible_identities(P2_vector):
        """Returns the possible identities by the argmax of the P2 vector and
        the value of the maximum.
        """
        max = np.max(P2_vector)  # there can be two equal maximums
        return np.argwhere(P2_vector == max)[:, 0] + 1, max

    def assign_identity(self, number_of_animals: int):
        """Assigns the identity to the fragment by considering the fragments
        coexisting with it.

        If the certainty of the identification is high enough it sets
        the identity of the fragment as fixed and it won't be modified during
        the postprocessing.
        """
        assert self.is_an_individual
        if self.identity_is_fixed:
            return
        if self.used_for_training:
            self.identity_is_fixed = True
            return

        possible_identities, max_P2 = self.get_possible_identities(self.P2_vector)
        if len(possible_identities) > 1:  # TODO is it possible?
            self.identity = 0
            self.zero_identity_assigned_by_P2 = True
            self.ambiguous_identities = possible_identities
        else:
            if max_P2 > conf.FIXED_IDENTITY_THRESHOLD:
                self.identity_is_fixed = True
            self.identity = possible_identities[0]
            self.P1_vector = np.zeros(len(self.P1_vector))
            self.P1_vector[self.identity - 1] = 1.0
            for fragment in self.coexisting_individual_fragments:
                fragment.compute_P2_vector(number_of_animals)

    def compute_P2_vector(self, number_of_animals: int):
        """Computes the P2_vector of the fragment.

        It is based on :attr:`coexisting_individual_fragments`"""
        coexisting_P1_vectors = np.asarray(
            [fragment.P1_vector for fragment in self.coexisting_individual_fragments]
        )
        numerator = self.P1_vector * np.prod(1.0 - coexisting_P1_vectors, axis=0)
        denominator = numerator.sum()
        if denominator != 0:
            self.P2_vector = numerator / denominator
        else:
            self.P2_vector = np.zeros(number_of_animals)

    @property
    def certainty_P2(self) -> float:
        """Indicating the certainty of the identity following the P2"""

        if self.P2_vector is None or self.P2_vector.sum() < 0.001:
            return 0.0

        P2_vector_ordered = np.sort(self.P2_vector)
        P2_first_max = P2_vector_ordered[-1]
        P2_second_max = P2_vector_ordered[-2]

        with np.errstate(divide="ignore"):
            return P2_first_max / P2_second_max

    @staticmethod
    def compute_P1_from_frequencies(frequencies: np.ndarray):
        """Given the frequencies of a individual fragment
        computer the P1 vector.

        P1 is the softmax of the frequencies with base 2 for each identity.
        Numpy array indicating the number of images assigned with each of
        the possible identities
        """
        with np.errstate(over="ignore"):
            return 1.0 / (
                2.0
                ** (
                    np.tile(frequencies, (len(frequencies), 1)).T
                    - np.tile(frequencies, (len(frequencies), 1))
                )
            ).sum(axis=0)

    @staticmethod
    def compute_median_softmax(softmax_probs, number_of_animals):
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
        softmax_probs = np.asarray(softmax_probs)
        # jumps are fragment composed by a single image, thus:
        if len(softmax_probs.shape) == 1:
            softmax_probs = np.expand_dims(softmax_probs, axis=1)
        max_softmax_probs = np.max(softmax_probs, axis=1)
        argmax_softmax_probs = np.argmax(softmax_probs, axis=1)
        softmax_median = np.zeros(number_of_animals)
        for i in np.unique(argmax_softmax_probs):
            softmax_median[i] = np.median(max_softmax_probs[argmax_softmax_probs == i])
        return softmax_median

    @staticmethod
    def compute_certainty_of_individual_fragment(
        P1_vector: np.ndarray, median_softmax
    ) -> float:
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
        argsort_p1_vector = P1_vector.argsort()
        sorted_p1_vector = P1_vector[argsort_p1_vector]
        sorted_softmax_probs = median_softmax[argsort_p1_vector]
        certainty = (
            np.diff((sorted_p1_vector * sorted_softmax_probs)[-2:])
            / sorted_p1_vector[-2:].sum()
        )
        return certainty[0]

    def get_neighbour_fragment(
        self,
        fragments: list["Fragment"],
        scope: str,
        number_of_frames_in_direction: int = 0,
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
        # TODO optimize
        if scope == "to_the_past":
            neighbour = [
                fragment
                for fragment in fragments
                if fragment.is_an_individual
                and len(fragment.assigned_identities) == 1
                and fragment.assigned_identities[0] == self.assigned_identities[0]
                and self.start_frame - fragment.end_frame
                == number_of_frames_in_direction
            ]
        elif scope == "to_the_future":
            neighbour = [
                fragment
                for fragment in fragments
                if fragment.is_an_individual
                and len(fragment.assigned_identities) == 1
                and fragment.assigned_identities[0] == self.assigned_identities[0]
                and fragment.start_frame - self.end_frame
                == number_of_frames_in_direction
            ]
        else:
            raise ValueError(scope)

        assert len(neighbour) < 2
        return neighbour[0] if len(neighbour) == 1 else None

    def set_partially_or_globally_accumulated(self, accumulation_strategy):
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
            f"Is certain: {self.is_certain}",
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
