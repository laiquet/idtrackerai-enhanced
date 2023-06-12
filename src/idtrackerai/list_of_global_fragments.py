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
import json
import logging
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from . import Blob, Fragment, GlobalFragment
from .utils import conf, resolve_path


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

    non_accumulable_global_fragments: list[GlobalFragment]
    """List of global fragments which are NOT candidate for accumulation"""

    global_fragments: list[GlobalFragment]
    """List of global fragments which are candidate for accumulation"""

    def __init__(self, global_fragments: Iterable[GlobalFragment]):
        self.global_fragments = []
        self.non_accumulable_global_fragments = []

        for global_fragment in global_fragments:
            if (
                global_fragment.min_n_images_per_fragment
                > conf.MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
            ):
                self.global_fragments.append(global_fragment)
            else:
                self.non_accumulable_global_fragments.append(global_fragment)

        logging.info(
            "Total number of global_fragments: %d",
            len(self.non_accumulable_global_fragments) + len(self.global_fragments),
        )
        logging.info(
            "Of which %d are long enough to be accumulated", len(self.global_fragments)
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
        global_fragments_boolean_array = [
            is_global_fragment_core(blobs_in_frame, blobs_in_video[i - 1], num_animals)
            for i, blobs_in_frame in enumerate(blobs_in_video)
        ]

        indices_beginning_of_fragment = detect_global_fragments_core_first_frame(
            global_fragments_boolean_array
        )

        return cls(
            GlobalFragment(blobs_in_video, fragments, i)
            for i in indices_beginning_of_fragment
        )

    @property
    def single_global_fragment(self) -> bool:
        return len(self.global_fragments) == 1

    @property
    def no_global_fragment(self) -> bool:
        return len(self.global_fragments) == 0

    def sort_by_distance_travelled(self):
        self.global_fragments.sort(
            key=lambda x: x.minimum_distance_travelled, reverse=True
        )

    def order_by_distance_to_the_frame(self, frame_number: int):
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
        self.global_fragments.sort(
            key=lambda x: abs(x.first_frame_of_the_core - frame_number)
        )

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
        path.parent.mkdir(exist_ok=True)

        json.dump(self.__dict__, path.open("w"), cls=GlobalFragmentsEncoder, indent=4)

    @classmethod
    def load(
        cls, path: Path | str, fragments: list[Fragment] | None = None
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

        if not path.is_file():  # <=5.1.3 compatibility
            if not path.with_suffix(".pickle").is_file():
                raise FileNotFoundError(path)
            pickle.load(path.with_suffix(".pickle").open("rb")).save(path)

        list_of_global_fragments = cls.__new__(cls)
        json_data = json.load(path.open("r"))

        list_of_global_fragments.global_fragments = [
            GlobalFragment.from_json(g_frag_data, fragments)
            for g_frag_data in json_data["global_fragments"]
        ]

        list_of_global_fragments.non_accumulable_global_fragments = [
            GlobalFragment.from_json(g_frag_data, fragments)
            for g_frag_data in json_data["non_accumulable_global_fragments"]
        ]

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
    ]  # boolean_array[0] is always False


def is_global_fragment_core(
    blobs_in_frame: list[Blob], blobs_in_frame_past: list[Blob], n_animals: int
) -> bool:
    """Return True if the set of fragments identifiers in the current frame
    is the same as in the previous frame, otherwise returns false
    """
    all_in_frame = len(blobs_in_frame) == n_animals

    same_fragment_identifiers = {b.fragment_identifier for b in blobs_in_frame} == {
        b.fragment_identifier for b in blobs_in_frame_past
    }
    not_all_were_in_frame = len(blobs_in_frame_past) != n_animals
    return all_in_frame and (same_fragment_identifiers or not_all_were_in_frame)


class GlobalFragmentsEncoder(json.JSONEncoder):
    """Json encoder to serialize Global Fragments with styled indentation"""

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)

        if isinstance(obj, GlobalFragment):
            serial = obj.__dict__.copy()
            serial.pop("individual_fragments", None)  # remove connections

            serial["individual_fragments_identifiers"] = (  # without indentation
                f"NotString{json.dumps(obj.individual_fragments_identifiers)}"
            )

            return serial

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        return super().default(obj)

    def iterencode(self, obj, **kwargs):
        for encoded in super().iterencode(obj, **kwargs):
            if encoded.startswith('"NotString'):
                # remove colons and "NotString"
                yield encoded[10:-1]
            else:
                yield encoded
