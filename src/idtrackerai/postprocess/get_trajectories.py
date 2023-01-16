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
from importlib import metadata

import numpy as np
from rich.progress import track

from idtrackerai import Blob, Video
from idtrackerai.utils import conf


def produce_trajectories(blobs_in_video: list[list[Blob]], number_of_animals: int):
    """Produce trajectories array from ListOfBlobs

    Parameters
    ----------
    blobs_in_video : <ListOfBlobs object>
        See :class:`list_of_blobs.ListOfBlobs`
    number_of_frames : int
        Total number of frames in video
    number_of_animals : int
        Number of animals to be tracked

    Returns
    -------
    dict
        Dictionary with np.array as values (trajectories organized by identity)

    """
    number_of_frames = len(blobs_in_video)
    centroid_trajectories = np.full((number_of_frames, number_of_animals, 2), np.NaN)
    id_probabilities = np.full((number_of_frames, number_of_animals, 1), np.NaN)

    areas = np.full((number_of_frames, number_of_animals), np.NaN)

    for blobs_in_frame in track(blobs_in_video, description="Producing trajectories"):
        for blob in blobs_in_frame:
            for identity, centroid in zip(blob.final_identities, blob.final_centroids):
                if identity not in (None, 0):
                    centroid_trajectories[blob.frame_number, identity - 1, :] = centroid
            if (
                blob.is_an_individual
                and len(blob.final_identities) == 1
                and blob.P2_vector is not None
            ):
                identity = blob.final_identities[0]
                if identity not in (None, 0):
                    id_probabilities[blob.frame_number, identity - 1, :] = np.max(
                        blob.P2_vector
                    )
                    areas[blob.frame_number, identity - 1] = blob.area

    trajectories_info_dict = {
        "centroid_trajectories": centroid_trajectories,
        "id_probabilities": id_probabilities,
        "areas": areas,
    }
    return trajectories_info_dict


def produce_trajectories_wo_identification(
    blobs_in_video: list[list[Blob]], number_of_animals: int
):
    number_of_frames = len(blobs_in_video)
    centroid_trajectories = np.full((number_of_frames, number_of_animals, 2), np.nan)
    identifiers_prev = np.full(number_of_animals, np.nan)

    areas = np.full((number_of_frames, number_of_animals), np.nan)

    for frame_number, blobs_in_frame in track(
        enumerate(blobs_in_video), "Creating trajectories"
    ):
        try:
            identifiers_next = set(
                b.fragment_identifier for b in blobs_in_video[frame_number + 1]
            )
        except IndexError:  # last frame
            identifiers_next = set(b.fragment_identifier for b in blobs_in_frame)

        for blob in blobs_in_frame:
            if blob.is_an_individual:
                if blob.fragment_identifier in identifiers_prev:
                    column = np.argwhere(identifiers_prev == blob.fragment_identifier)[
                        0
                    ][0]
                else:
                    column = np.argwhere(np.isnan(identifiers_prev))[0][0]
                    identifiers_prev[column] = blob.fragment_identifier

                blob.identity = column + 1
                # blobs that are individual only have one centroid
                centroid_trajectories[frame_number, column, :] = blob.final_centroids[0]
                areas[frame_number, column] = blob.area

                if blob.fragment_identifier not in identifiers_next:
                    identifiers_prev[column] = np.nan
    trajectories_info_dict = {
        "centroid_trajectories": centroid_trajectories,
        "id_probabilities": None,
        "areas": areas,
    }
    return trajectories_info_dict


def produce_output_dict(blobs_in_video: list[list[Blob]], video: Video):
    """Outputs the dictionary with keys: trajectories, git_commit, video_path,
    frames_per_second

    Parameters
    ----------
    blobs_in_video : list
        List of all blob objects (see :class:`~blob.Blobs`) generated by
        considering all the blobs segmented from the video
    video : <Video object>
        See :class:`~video.Video`

    Returns
    -------
    dict
        Output dictionary containing trajectories as values

    """
    assert len(blobs_in_video) == video.number_of_frames
    if video.track_wo_identities:
        video.number_of_animals = max(len(bf) for bf in blobs_in_video)
        trajectories_info_dict = produce_trajectories_wo_identification(
            blobs_in_video, video.number_of_animals
        )
    else:
        trajectories_info_dict = produce_trajectories(
            blobs_in_video, video.number_of_animals
        )

    output_dict = {
        "trajectories": trajectories_info_dict["centroid_trajectories"]
        / video.resolution_reduction,
        "version": metadata.version("idtrackerai"),
        "video_paths": video.video_paths,
        "frames_per_second": video.frames_per_second,
        "body_length": video.median_body_length_full_resolution,
        "stats": {"estimated_accuracy": video.estimated_accuracy},
    }

    if trajectories_info_dict["id_probabilities"] is not None:
        output_dict["id_probabilities"] = trajectories_info_dict["id_probabilities"]
        # After the interpolation some identities that were 0 are assigned
        output_dict["stats"]["estimated_accuracy_after_interpolation"] = (
            1 if video.single_animal else np.nanmean(output_dict["id_probabilities"])
        )
        # Centroids with identity
        identified = ~np.isnan(output_dict["trajectories"][..., 0])
        output_dict["stats"]["percentage_identified"] = np.mean(identified)
        # Estimated accuracy of identified blobs

        output_dict["stats"]["estimated_accuracy_identified"] = (
            1
            if video.single_animal
            else np.nanmean(output_dict["id_probabilities"][identified])
        )

    if conf.SAVE_AREAS:
        output_dict["areas"] = trajectories_info_dict["areas"]

    output_dict["setup_points"] = video.setup_points
    # This is only used in the validationGUI
    if hasattr(video, "identities_groups"):
        output_dict["identities_groups"] = video.identities_groups

    return output_dict
