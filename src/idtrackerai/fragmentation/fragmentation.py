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
from pathlib import Path

from rich.progress import track

from idtrackerai import Blob, ListOfBlobs, ListOfFragments
from idtrackerai.utils.py_utils import Timer


def fragmentation(
    list_of_blobs: ListOfBlobs,
    number_of_animals: int,
    id_images_file_paths: list[Path],
    timer: Timer,
) -> ListOfFragments | None:
    timer.tic()
    if number_of_animals != 1:
        if not list_of_blobs.blobs_are_connected:
            # If the list of of blobs has been loaded
            logging.warning("ListOfBlobs not connected, reconnecting now")
            list_of_blobs.compute_overlapping_between_subsequent_frames()

        number_of_individual_fragments = (
            compute_fragment_identifier_and_blob_index(
                list_of_blobs.blobs_in_video,
                max(
                    number_of_animals,
                    list_of_blobs.maximum_number_of_blobs,
                ),
            )
        )
        compute_crossing_fragment_identifier(
            list_of_blobs.blobs_in_video,
            number_of_individual_fragments,
        )

        # List of fragments
        list_of_fragments = ListOfFragments.from_fragmented_blobs(
            list_of_blobs.blobs_in_video,
            number_of_animals,
            id_images_file_paths,
        )
    else:
        # If there is only one animal there is no need to compute fragments
        # as the trajectories are obtained directly from the list_of_blobs
        list_of_fragments = None
    timer.tac()
    return list_of_fragments


def compute_fragment_identifier_and_blob_index(
    blobs_in_video: list[list[Blob]], number_of_animals: int
) -> int:
    """Associates a unique fragment identifier to individual blobs
    connected with its next and previous blobs.

    Blobs must be connected and classified as individuals or crossings.

    Parameters
    ----------
    number_of_animals : int
        Number of animals to be tracked as defined by the user
    """
    counter = 0
    possible_blob_indices = range(number_of_animals)
    set_possible_blob_indices = set(possible_blob_indices)

    for blobs_in_frame in track(
        blobs_in_video, description="Assigning fragment identifier"
    ):
        used_blob_indices = [
            blob.blob_index
            for blob in blobs_in_frame
            if blob.blob_index is not None
        ]
        missing_blob_indices = list(
            set_possible_blob_indices.difference(set(used_blob_indices))
        )
        for blob in blobs_in_frame:
            if blob.fragment_identifier is None and blob.is_an_individual:
                blob._fragment_identifier = counter
                blob_index = missing_blob_indices.pop(0)
                blob._blob_index = blob_index
                if (
                    len(blob.next) == 1
                    and len(blob.next[0].previous) == 1
                    and blob.next[0].is_an_individual
                ):
                    blob.next[0]._fragment_identifier = counter
                    blob.next[0]._blob_index = blob_index
                    if blob.next[0].is_an_individual_in_a_fragment:
                        blob = blob.next[0]

                        while (
                            len(blob.next) == 1
                            and blob.next[0].is_an_individual_in_a_fragment
                        ):
                            blob = blob.next[0]
                            blob._fragment_identifier = counter
                            blob._blob_index = blob_index

                        if (
                            len(blob.next) == 1
                            and len(blob.next[0].previous) == 1
                            and blob.next[0].is_an_individual
                        ):
                            blob.next[0]._fragment_identifier = counter
                            blob.next[0]._blob_index = blob_index
                counter += 1
    logging.info(f"{counter} individual fragments")
    return counter


# TODO: This is part of fragmentation it should be somewhere else.
def compute_crossing_fragment_identifier(
    blobs_in_video: list[list[Blob]], number_of_individual_fragments: int
):
    """Assign a unique identifier to fragments associated to crossing
    blobs.

    Fragment identifiers of crossings fragments start from the last
    fragment identifier of the individual fragments.
    """

    fragment_identifier = number_of_individual_fragments

    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_crossing and blob.fragment_identifier is None:
                blob._fragment_identifier = fragment_identifier
                cur_blob = blob

                while (
                    len(cur_blob.next) == 1
                    and len(cur_blob.next[0].previous) == 1
                    and cur_blob.next[0].is_a_crossing
                ):
                    cur_blob = cur_blob.next[0]
                    cur_blob._fragment_identifier = fragment_identifier

                cur_blob = blob

                while (
                    len(cur_blob.previous) == 1
                    and len(cur_blob.previous[0].next) == 1
                    and cur_blob.previous[0].is_a_crossing
                ):
                    cur_blob = cur_blob.previous[0]
                    cur_blob._fragment_identifier = fragment_identifier

                fragment_identifier += 1
    logging.info(
        f"{fragment_identifier - number_of_individual_fragments} crossing fragments"
    )
    logging.info(f"{fragment_identifier} number of fragments in total")
