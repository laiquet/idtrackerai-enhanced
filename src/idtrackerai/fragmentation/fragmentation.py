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

from idtrackerai import Blob, ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Video
from idtrackerai.utils import track


def fragmentation_API(
    video: Video, list_of_blobs: ListOfBlobs
) -> tuple[ListOfFragments, ListOfGlobalFragments]:
    video.fragmentation_timer.start()

    compute_fragment_identifier_and_blob_index(
        list_of_blobs.blobs_in_video,
        max(video.number_of_animals, list_of_blobs.maximum_number_of_blobs),
    )

    list_of_fragments = ListOfFragments.from_fragmented_blobs(
        list_of_blobs.all_blobs, video.number_of_animals, video.id_images_file_paths
    )
    logging.info(
        f"{list_of_fragments.number_of_fragments} Fragments in total, "
        f"{list_of_fragments.number_of_individual_fragments} individuals and "
        f"{list_of_fragments.number_of_crossing_fragments} crossings"
    )

    list_of_global_fragments = ListOfGlobalFragments.from_fragments(
        list_of_blobs.blobs_in_video,
        list_of_fragments.fragments,
        video.number_of_animals,
    )
    list_of_fragments.manage_accumulable_non_accumulable_fragments(
        list_of_global_fragments.global_fragments,
        list_of_global_fragments.non_accumulable_global_fragments,
    )

    video.fragmentation_timer.finish()
    return list_of_fragments, list_of_global_fragments


def compute_fragment_identifier_and_blob_index(
    blobs_in_video: list[list[Blob]], number_of_animals: int
) -> None:
    """Associates a unique fragment identifier to individual blobs
    connected with its next and previous blobs.

    Blobs must be connected and classified as individuals or crossings.

    Parameters
    ----------
    number_of_animals : int
        Number of animals to be tracked as defined by the user
    """
    frame_id = 0
    possible_blob_indices = set(range(number_of_animals))

    for blobs_in_frame in track(blobs_in_video, "Fragmenting blobs"):
        missing_blob_indices = possible_blob_indices.difference(
            blob.blob_index for blob in blobs_in_frame
        )

        for blob in blobs_in_frame:
            if blob.fragment_identifier != -1:
                continue

            blob.fragment_identifier = frame_id
            if blob.is_an_individual:
                blob_index = missing_blob_indices.pop()
                blob.blob_index = blob_index
                while (
                    blob.n_next == 1
                    and blob.next[0].n_previous == 1
                    and blob.next[0].is_an_individual
                ):
                    blob = blob.next[0]
                    blob.fragment_identifier = frame_id
                    blob.blob_index = blob_index

            elif blob.is_a_crossing:
                while (
                    blob.n_next == 1
                    and blob.next[0].n_previous == 1
                    and blob.next[0].is_a_crossing
                ):
                    blob = blob.next[0]
                    blob.fragment_identifier = frame_id

            frame_id += 1
