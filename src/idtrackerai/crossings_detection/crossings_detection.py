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
from idtrackerai import ListOfBlobs, Video
from idtrackerai.utils import create_dir

from .crossing_detector import detect_crossings
from .model_area import ModelArea, compute_body_length


def crossings_detection_API(video: Video, list_of_blobs: ListOfBlobs) -> None:
    """
    This crossings detector works under the following assumptions
        1. The number of animals in the video is known (given by the user)
        2. There are frames in the video where all animals are separated from
        each other.
        3. All animals have a similar size
        4. The frame rate of the video is higher enough so that consecutive
        segmented blobs of pixels of the same animal overlap, i.e. some of the
        pixels representing the animal A in frame i are the same in the
        frame i+1.

    NOTE: This crossing detector sets the identification images that will be
    used to identify the animals
    """
    video.crossing_detector_timer.start()

    median_body_length = compute_body_length(
        list_of_blobs.blobs_in_video, video.number_of_animals
    )
    video.set_id_image_size(median_body_length)

    create_dir(video.id_images_folder, remove_existing=True)

    list_of_blobs.set_images_for_identification(
        video.episodes,
        video.id_images_file_paths,
        video.id_image_size,
        video.segmentation_data_folder,
    )
    list_of_blobs.compute_overlapping_between_subsequent_frames()
    create_dir(video.crossings_detector_folder)

    if not video.single_animal:
        model_area = ModelArea(list_of_blobs.blobs_in_video, video.number_of_animals)
        detect_crossings(list_of_blobs, video, model_area)
    else:
        for blob in list_of_blobs.all_blobs:
            blob.is_an_individual = True

    video.crossing_detector_timer.finish()
