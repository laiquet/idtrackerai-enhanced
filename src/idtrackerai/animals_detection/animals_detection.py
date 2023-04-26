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

import cv2

from idtrackerai import ListOfBlobs, Video
from idtrackerai.utils import CustomError, create_dir

from .segmentation import compute_background, segment


def animals_detection_API(video: Video):
    """
    This class generates a ListOfBlobs object and updates the video
    object with information about the process.

    Parameters
    ----------
    video: Video
        An instance of the class :class:`~idtrackerai.video.Video`.

    Attributes
    ----------
    video: Video
    list_of_blobs: ListOfBlobs
    detection_parameters: Dict

    See Also
    --------
    :class:`~idtrackerai.list_of_blobs.ListOfBlobs`
    """
    video.detect_animals_timer.start()
    create_dir(video.segmentation_data_folder, remove_existing=True)

    bkg_model = video.bkg_model
    if video.use_bkg:
        if bkg_model is None:
            bkg_model = compute_background(
                video.video_paths, video.ROI_mask, video.episodes
            )
            video.bkg_model = bkg_model
        else:
            logging.info("Using previously computed background model from GUI")
    else:
        bkg_model = None
        logging.info("No background model computed")

    detection_parameters = {
        "intensity_ths": video.intensity_ths,
        "area_ths": video.area_ths,
        "ROI_mask": video.ROI_mask,
        "bkg_model": bkg_model,
        "resolution_reduction": video.resolution_reduction,
    }

    if video.resolution_reduction != 1 and bkg_model is not None:
        detection_parameters["bkg_model"] = cv2.resize(
            bkg_model,
            None,  # type: ignore
            fx=video.resolution_reduction,
            fy=video.resolution_reduction,
            interpolation=cv2.INTER_AREA,
        )

    # Main call
    blobs_in_video = segment(
        detection_parameters,
        video.episodes,
        video.segmentation_data_folder / "blobs_bbox_images.hdf5",
        video.video_paths,
        video.number_of_frames,
    )

    list_of_blobs = ListOfBlobs(blobs_in_video)
    assert len(list_of_blobs) == video.number_of_frames
    logging.info(f"{list_of_blobs.number_of_blobs} detected blobs in total")

    if video.number_of_animals > 0:
        check_segmentation(video, list_of_blobs)

    video.detect_animals_timer.finish()
    return list_of_blobs


def check_segmentation(video: Video, list_of_blobs: ListOfBlobs):
    """
    idtracker.ai is designed to work under the assumption that all the
    detected blobs are animals. In the frames where the number of
    detected blobs is higher than the number of animals in the video, it is
    likely that some blobs do not represent animals. In this scenario
    idtracker.ai might misbehave. This method allows to check such
    condition.
    """
    logging.info("Checking segmentation")

    n_frames_with_all_visible = sum(
        len(blobs) == video.number_of_animals for blobs in list_of_blobs.blobs_in_video
    )

    if n_frames_with_all_visible == 0:
        raise CustomError(
            "There is not any frames where the number of blobs is equal "
            "to the number of animals stated by the user. Idtracker.ai "
            "needs those frame to work"
        )

    error_frames = [
        frame
        for frame, blobs in enumerate(list_of_blobs.blobs_in_video)
        if len(blobs) > video.number_of_animals
    ]

    n_error_frames = len(error_frames)
    logging.log(
        logging.WARNING if n_error_frames else logging.INFO,
        f"There are {n_error_frames} frames with more blobs than animals",
    )
    video.number_of_error_frames = n_error_frames

    if n_error_frames:
        logging.warning(
            "This can be detrimental for the proper functioning of the system"
        )
        if n_error_frames < 25:
            logging.warning(f"Frames with more blobs than animals: {error_frames}")
        else:
            logging.warning(
                "Too many frames with more blobs than animals "
                "for printing their indices in log"
            )

        output_path = video.session_folder / "inconsistent_frames.csv"
        logging.info(
            f"Saving indices of frames with more blobs than animals in {output_path}"
        )
        output_path.write_text("\n".join(map(str, error_frames)))

        if video.check_segmentation:
            list_of_blobs.save(video.blobs_path)
            raise CustomError(
                f"Check_segmentation is {True}, exiting...\n"
                "Please readjust the segmentation parameters and track again"
            )
        logging.info(f"Check_segmentation is {False}, ignoring the above errors")
