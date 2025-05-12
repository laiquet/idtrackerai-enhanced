import json
import logging
import pickle
from collections.abc import Iterable, Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from . import Blob, Fragment, GlobalFragment, conf
from .utils import deprecated, resolve_path


class GlobalFragmentsEncoder(json.JSONEncoder):
    """Json encoder to serialize Global Fragments with styled indentation"""

    def default(self, o) -> Any:
        match o:
            case set():
                return list(o)
            case GlobalFragment():
                serial = o.__dict__.copy()
                serial.pop("fragments", None)  # remove connections

                serial["fragments_identifiers"] = (  # without indentation
                    f"NotString{json.dumps(o.fragments_identifiers)}"
                )

                return serial
            case np.integer():
                return int(o)
            case np.floating():
                return float(o)
            case _:
                return super().default(o)

    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
        return (
            encoded[10:-1] if encoded.startswith('"NotString') else encoded
            for encoded in super().iterencode(o, _one_shot)
        )


class ListOfGlobalFragments:
    """Contains all the instances of the class :class:`GlobalFragment`.

    It contains methods to retrieve information from these global fragments
    and to update their attributes.
    These methods are mainly used during the cascade of training and
    identification protocols.

    Parameters
    ----------
    global_fragments : list
        List of instances of :class:`GlobalFragment`.
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
                >= conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
            ):
                self.global_fragments.append(global_fragment)
            else:
                self.non_accumulable_global_fragments.append(global_fragment)

        logging.info(
            "Total number of Global Fragments: %d",
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
    ) -> "ListOfGlobalFragments":
        """Creates the list of instances of the class :class:`GlobalFragment`.

        Parameters
        ----------
        blobs_in_video : list[list[Blob]]
            All blobs in video
        fragments : list[Fragment]
            All Fragments in video
        num_animals : int
            Number of animals to be tracked as indicated by the user.

        Returns
        -------
        ListOfGlobalFragments
            The list of all Global Fragments
        """
        is_global_fragment_core = [False] + [
            get_global_fragment_core(blobs_in_frame, blobs_in_past, num_animals)
            for blobs_in_past, blobs_in_frame in pairwise(blobs_in_video)
        ]

        global_fragments_first_frames = [
            frame
            for frame, (was_core, is_core) in enumerate(
                pairwise(is_global_fragment_core), 1
            )
            if is_core and not was_core
        ]

        return cls(
            GlobalFragment(
                [fragments[blob.fragment_identifier] for blob in blobs_in_video[i]]
            )
            for i in global_fragments_first_frames
        )

    def __len__(self) -> int:
        return len(self.global_fragments)

    def __iter__(self) -> Iterator[GlobalFragment]:
        return iter(self.global_fragments)

    def sort_by_distance_travelled(self):
        self.global_fragments.sort(
            key=lambda x: x.minimum_distance_travelled, reverse=True
        )

    def sort_by_distance_to_the_frame(self, frame_number: int):
        """Sorts the global fragments with respect to their distance from the
        first global fragment chose for accumulation.
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
        path : Path | str
            Path where the object will be stored
        """
        path = resolve_path(path)
        logging.info(f"Saving ListOfGlobalFragments at {path}", stacklevel=2)
        path.parent.mkdir(exist_ok=True)

        json.dump(self.__dict__, path.open("w"), cls=GlobalFragmentsEncoder, indent=4)

    @classmethod
    def load(
        cls, path: Path | str, fragments: list[Fragment] | None = None
    ) -> "ListOfGlobalFragments":
        """Loads an instance of the class saved with :meth:`save`

        Parameters
        ----------

        path : str
            Path where the object to be loaded is stored.
        fragments : list
            List of all the instances of the class :class:`Fragment`
            in the video.
        """
        path = resolve_path(path)
        logging.info(f"Loading ListOfGlobalFragments from {path}", stacklevel=2)

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

    @deprecated(
        version="6.0.0",
        reason="Use `len(ListOfGlobalFragments.global_fragments) == 1` instead",
    )
    def single_global_fragment(self) -> bool:
        """Returns True if there is only one global fragment in the list."""
        return len(self.global_fragments) == 1

    @deprecated(
        version="6.0.0",
        reason="Use `len(ListOfGlobalFragments.global_fragments) == 0` instead",
    )
    def no_global_fragment(self) -> bool:
        """Returns True if there are no global fragments in the list."""
        return len(self.global_fragments) == 0


def get_global_fragment_core(
    blobs_in_frame: list[Blob], blobs_in_past_frame: list[Blob], n_animals: int
) -> bool:
    """Return True if the frame belongs to a Global Fragment core.

    The last condition in the return line is there because we will look for
    [..., False, True, ...] patterns to get the first frame of the Global Fragment core.
    So we set the frame i to False if the previous frame is in a different
    Global Fragment (which means was_clean_frame and same_fragment_identifiers = False).
    """
    if n_animals == 0:  # unknown number of animals
        return False
    is_clean_frame = len(blobs_in_frame) == n_animals and all(
        b.is_an_individual for b in blobs_in_frame
    )
    was_clean_frame = len(blobs_in_past_frame) == n_animals and all(
        b.is_an_individual for b in blobs_in_past_frame
    )

    same_fragment_identifiers = {b.fragment_identifier for b in blobs_in_frame} == {
        b.fragment_identifier for b in blobs_in_past_frame
    }
    return is_clean_frame and (same_fragment_identifiers or not was_clean_frame)
