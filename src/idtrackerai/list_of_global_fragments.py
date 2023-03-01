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
import logging
import pickle
from itertools import chain
from pathlib import Path

from idtrackerai import Blob, Fragment, GlobalFragment
from idtrackerai.utils import resolve_path


class ListOfGlobalFragments:
    """Contains a list of instances of the class
    :class:`global_fragment.GlobalFragment`.

    It contains methods to retrieve information from these global fragments
    and to update their attributes.
    These methods are manily used during the cascade of training and
    identification protocols.

    Parameters
    ----------
    global_fragments : list
        List of instances of :class:`global_fragment.GlobalFragment`.
    """

    accumulation_step: int | None = None
    """Integer indicating the accumulation step at which the fragment was
    accumulated. See also the accumulation_manager.py module."""

    def __init__(self, global_fragments: list[GlobalFragment]):
        self.non_accumulable_global_fragments: list[GlobalFragment] = [
            global_fragment
            for global_fragment in global_fragments
            if not global_fragment.candidate_for_accumulation
        ]
        """List of global fragments which are NOT candidate for accumulation"""

        self.global_fragments: list[GlobalFragment] = [
            global_fragment
            for global_fragment in global_fragments
            if global_fragment.candidate_for_accumulation
        ]
        """List of global fragments which are candidate for accumulation"""

    @property
    def maximum_number_of_images(self) -> int:
        """The maximum number of images in the global fragment"""
        return max(
            global_fragment.total_number_of_images
            for global_fragment in self.global_fragments
        )

    @classmethod
    def from_fragments(
        cls,
        blobs_in_video: list[list[Blob]],
        fragments: list[Fragment],
        num_animals: int,
    ):
        """Creates the list of instances of the class
        :class:`~globalfragment.GlobalFragment`
        used to create :class:`.ListOfGlobalFragments`.

        Parameters
        ----------
        blobs_in_video : list
            List of lists with instances of the class class :class:`blob.Blob`).
        fragments : list
            List of instances of the class :class:`fragment.Fragment`
        num_animals : int
            Number of animals to be tracked as indicated by the user.

        Returns
        -------
        list
            list of instances of the class :class:`~globalfragment.GlobalFragment`

        """
        global_fragments_boolean_array = check_global_fragments(
            blobs_in_video, num_animals
        )
        indices_beginning_of_fragment = detect_global_fragments_core_first_frame(
            global_fragments_boolean_array
        )

        global_fragments = [
            GlobalFragment(blobs_in_video, fragments, i, num_animals)
            for i in indices_beginning_of_fragment
        ]
        logging.info(f"Total number of global_fragments: {len(global_fragments)}")
        return cls(global_fragments)

    @property
    def number_of_global_fragments(self) -> int:
        return len(self.global_fragments)

    @property
    def single_global_fragment(self) -> bool:
        return self.number_of_global_fragments == 1

    def reset(self, roll_back_to=None):
        """Resets all the global fragment by calling recursively the method
        :meth:`globalfragment.GlobalFragment.reset`.
        """
        logging.info(f"Resetting ListOfGlobalFragments to '{roll_back_to}'")
        for global_fragment in self.global_fragments:
            global_fragment.reset(roll_back_to)

    def order_by_distance_travelled(self):
        """Sorts the global fragments by the minimum distance travelled.
        See :attr:`global_fragment.GlobalFragment.minimum_distance_travelled`
        """
        self.global_fragments = sorted(
            self.global_fragments,
            key=lambda x: x.minimum_distance_travelled,
            reverse=True,
        )

    def set_first_global_fragment_for_accumulation(
        self, accumulation_trial: int
    ) -> GlobalFragment | None:
        """Sets the first global fragment that will be used during the
        accumulation in the cascade of training and identification protocols.

        If the user asked to perform identity transfer, then the identities
        of the first global fragment will be tried to be assigned using the
        neural network provided by the knowledge_transfer_folder parameter.

        Parameters
        ----------
        video : :class:`video.Video`
            Video object containing information about the video and the
            tracking process.
        accumulation_trial : int, optional
            Accumulation trials during the cascade of training and
            identification protocols, by default 0.
        identification_model : str, optional
            Path to the directory where the identification neural network
            model that should be used for identity transfer is stored,
            by default None.
        network_params : <NetworkParams object>, optional
            Object with the parameters of the network and how to train it,
             by default None.
        knowledge_transfer_info_dict : dic, optional
            Dictionary with information about the knowledge transfer,
            by default None.

        Returns
        -------
        int
            A unique identifier of the global fragment that will be used as the
            first global fragment for training.
        """
        self.order_by_distance_travelled()

        try:
            self.first_global_fragment_for_accumulation = self.global_fragments[
                accumulation_trial
            ]
        except IndexError:  # TODO what happens with this exception
            return None

        return self.first_global_fragment_for_accumulation

    def order_by_distance_to_the_first_global_fragment_for_accumulation(
        self, first_frame_first_global_fragment: list, accumulation_trial: int
    ):
        """Sorts the global fragments with respect to their distance from the
        first global fragment chose for accumulation.

        Parameters
        ----------
        video : :class:`video.Video`
            Instance of the class :class:`video.Video`.
        accumulation_trial : int
            accumulation number (protocol 2 performs a single accumulation
            attempt, and if used, protocol 3 will perform 3 other attempts)
        """
        self.global_fragments = sorted(
            self.global_fragments,
            key=lambda x: abs(
                x.first_frame_of_the_core
                - first_frame_first_global_fragment[accumulation_trial]
            ),
            reverse=False,
        )

    def relink_fragments_to_global_fragments(self, fragments: list[Fragment]):
        """Re-assigns the instances of :class:`fragment.Fragment` to each
        global fragment in the list of `global_fragments`.

        Parameters
        ----------
        fragments: list
            List of instances of the class :class:`fragment.Fragment`.
        """
        for global_fragment in self.global_fragments:
            global_fragment.set_individual_fragments(fragments)
        for global_fragment in self.non_accumulable_global_fragments:
            global_fragment.set_individual_fragments(fragments)

    def save(self, path: Path | str):
        """Saves an instance of the class.

        Before saving the instances of fragments associated to every global
        fragment are removed and reset them after saving. This
        prevents problems when pickling objects inside of objects.

        Parameters
        ----------
        global_fragments_path : str
            Path where the object will be stored
        """
        path = resolve_path(path)
        logging.info(f"Saving ListOfGlobalFragments at {path}")
        tmp_fragments = []
        for global_fragment in chain(
            self.global_fragments, self.non_accumulable_global_fragments
        ):
            tmp_fragments.append(global_fragment.individual_fragments)
            global_fragment.individual_fragments = []

        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

        for fragments, global_fragment in zip(
            tmp_fragments,
            chain(self.global_fragments, self.non_accumulable_global_fragments),
        ):
            global_fragment.individual_fragments = fragments

    @staticmethod
    def load(
        path: Path | str, fragments: list[Fragment] | None = None
    ) -> "ListOfGlobalFragments":
        """Loads an instance of the class saved with :meth:`save` and
        associates individual fragments to each global fragment by calling
        :meth:`~relink_fragments_to_global_fragments`

        Parameters
        ----------

        path_to_load : str
            Path where the object to be loaded is stored.
        fragments : list
            List of all the instances of the class :class:`fragment.Fragment`
            in the video.
        """
        path = resolve_path(path)
        logging.info(f"Loading ListOfGlobalFragments from {path}")
        with open(path, "rb") as file:
            list_of_global_fragments: ListOfGlobalFragments = pickle.load(file)

        if fragments is not None:
            list_of_global_fragments.relink_fragments_to_global_fragments(fragments)
        return list_of_global_fragments


def detect_global_fragments_core_first_frame(boolean_array: list[bool]) -> list[int]:
    """Detects the frame where the core of a global fragment starts.

    A core of a global fragment is the part of the global fragment where all
    the individuals are visible, i.e. the number of animals in the frame equals
    the number of animals in the video :boolean_array: array with True
    where the number of animals in the frame equals the number of animals in
    the video.
    """
    if all(boolean_array):
        return [0]
    return [
        i
        for i in range(len(boolean_array))
        if (boolean_array[i] and not boolean_array[i - 1])
    ]


def check_global_fragments(
    blobs_in_video: list[list[Blob]], num_animals: int
) -> list[bool]:
    """Returns list of booleans indicating the frames where all animals are
    visible.

    The element of the array is True if.

    * each blob has a unique blob intersecting in the past and future.
    * number of blobs equals num_animals.

    Parameters
    ----------
    blobs_in_video : list
        List of lists of instances of the class :class:`blob.Blob`.

    Returns
    -------
    list
        List of booleans with length the number of frames in the video. An
        element is True if all the animals are visible in the frame.
    """

    def _same_fragment_identifier(
        blobs_in_frame: list[Blob], blobs_in_frame_past: list[Blob]
    ) -> bool:
        """Return True if the set of fragments identifiers in the current frame
        is the same as in the previous frame, otherwise returns false
        """
        same_fragment_identifier = set(
            b.fragment_identifier for b in blobs_in_frame
        ) == set(b.fragment_identifier for b in blobs_in_frame_past)
        condition_2 = (
            all(b.is_an_individual for b in blobs_in_frame_past)
            and len(blobs_in_frame_past) == num_animals
        )
        return same_fragment_identifier or not condition_2

    return [
        all(b.is_an_individual for b in blobs_in_frame)
        and len(blobs_in_frame) == num_animals
        and _same_fragment_identifier(blobs_in_frame, blobs_in_video[i - 1])
        for i, blobs_in_frame in enumerate(blobs_in_video)
    ]
