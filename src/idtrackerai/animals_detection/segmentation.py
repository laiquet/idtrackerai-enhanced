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

from typing import Tuple, List, Dict, Optional
import logging
import multiprocessing
import os

import cv2
import h5py

from confapp import conf
from joblib import Parallel, delayed

from idtrackerai.blob import Blob
from idtrackerai.utils.py_utils import (
    set_mkl_to_multi_thread,
    set_mkl_to_single_thread,
)
from idtrackerai.animals_detection.segmentation_utils import (
    blob_extractor,
    get_frame_average_intensity,
    segment_frame,
    to_gray_scale,
    gaussian_blur,
)

logger = logging.getLogger("__main__.segmentation")

"""
The segmentation module
"""


def _get_blobs_in_frame(
    cap,
    video_params_to_store,
    segmentation_parameters,
    global_frame_number,
    frame_number_in_video_path,
    bounding_box_images_path,
    video_path,
    pixels_path,
    save_pixels,
    save_segmentation_image,
):
    """Segments a frame read from `cap` according to the preprocessing parameters
    in `video`. Returns a list `blobs_in_frame` with the Blob objects in the frame
    and the `max_number_of_blobs` found in the video so far. Frames are segmented
    in gray scale.

    Parameters
    ----------
    cap : <VideoCapture object>
        OpenCV object used to read the frames of the video
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    segmentation_thresholds : dict
        Dictionary with the thresholds used for the segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process
    frame_number : int
        Number of the frame being segmented. It is used to print in the terminal the frames
        where the segmentation fails. This frame is the frame of the episode if the video
        is chuncked.
    global_frame_number : int
        This is the frame number in the whole video. It will be different to the frame_number
        if the video is chuncked.


    Returns
    -------
    blobs_in_frame : list
        List of <Blob object> segmented in the current frame
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process

    See Also
    --------
    Video
    Blob
    segment_frame
    blob_extractor
    """
    ret, frame = cap.read()
    (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
    ) = _process_frame(
        frame,
        segmentation_parameters,
        global_frame_number,
        save_pixels,
        save_segmentation_image,
    )

    blobs_in_frame = _create_blobs_objects(
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
        save_segmentation_image,
        bounding_box_images_path,
        save_pixels,
        pixels_path,
        global_frame_number,
        frame_number_in_video_path,
        video_params_to_store,
        video_path,
        segmentation_parameters,
    )

    return blobs_in_frame


def _process_frame(
    frame,
    segmentation_parameters,
    frame_number,
    save_pixels,
    save_segmentation_image,
):

    try:
        frame = gaussian_blur(frame, sigma=conf.SIGMA_GAUSSIAN_BLURRING)
        bkg = segmentation_parameters["bkg_model"]
        mask = segmentation_parameters["mask"]
        # avg_brightness = segmentation_parameters["avg_brightness"]

        # Apply resolution reduction
        if segmentation_parameters["resolution_reduction"] != 1:
            frame = cv2.resize(
                frame,
                None,
                fx=segmentation_parameters["resolution_reduction"],
                fy=segmentation_parameters["resolution_reduction"],
                interpolation=cv2.INTER_AREA,
            )
            if bkg is not None:
                bkg = cv2.resize(
                    bkg,
                    None,
                    fx=segmentation_parameters["resolution_reduction"],
                    fy=segmentation_parameters["resolution_reduction"],
                    interpolation=cv2.INTER_AREA,
                )
            if mask is not None:
                mask = cv2.resize(
                    mask,
                    None,
                    fx=segmentation_parameters["resolution_reduction"],
                    fy=segmentation_parameters["resolution_reduction"],
                    interpolation=cv2.INTER_AREA,
                )
        # Convert the frame to gray scale
        gray = to_gray_scale(frame)
        # Normalize frame
        # flickering_factor = avg_brightness / get_frame_average_intensity(
        #     gray, mask
        # )
        # normalized_framed = cv2.convertScaleAbs(gray, alpha=flickering_factor)
        normalized_framed = gray / get_frame_average_intensity(gray, mask)
        # Binarize frame
        segmentedFrame = segment_frame(
            normalized_framed,
            segmentation_parameters["min_threshold"],
            segmentation_parameters["max_threshold"],
            bkg,
            mask,
            segmentation_parameters["subtract_bkg"],
        )

        # Extract blobs info
        (
            bounding_boxes,
            miniframes,
            centroids,
            areas,
            pixels,
            contours,
            estimated_body_lengths,
        ) = blob_extractor(
            segmentedFrame,
            gray,
            segmentation_parameters["min_area"],
            segmentation_parameters["max_area"],
            save_pixels,
            save_segmentation_image,
        )
    except Exception as e:
        print(f"Frame {frame_number}: {e}")
        logger.info(
            "An error occurred while reading frame number : %i" % frame_number
        )
        bounding_boxes = []
        miniframes = []
        centroids = []
        areas = []
        pixels = []
        contours = []
        estimated_body_lengths = []

    return (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
    )


def _create_blobs_objects(
    bounding_boxes,
    miniframes,
    centroids,
    areas,
    pixels,
    contours,
    estimated_body_lengths,
    save_segmentation_image,
    bounding_box_images_path,
    save_pixels,
    pixels_path,
    global_frame_number,
    frame_number_in_video_path,
    video_params_to_store,
    video_path,
    segmentation_parameters,
):
    blobs_in_frame = []
    # create blob objects
    for i, bounding_box in enumerate(bounding_boxes):
        if save_segmentation_image == "DISK":
            with h5py.File(bounding_box_images_path, "a") as f1:
                f1.create_dataset(
                    str(global_frame_number) + "-" + str(i), data=miniframes[i]
                )
            miniframes[i] = None
        if save_pixels == "DISK":
            with h5py.File(pixels_path, "a") as f2:
                f2.create_dataset(
                    str(global_frame_number) + "-" + str(i), data=pixels[i]
                )
            pixels[i] = None

        blob = Blob(
            centroids[i],
            contours[i],
            areas[i],
            bounding_box,
            bounding_box_image=miniframes[i],
            bounding_box_images_path=bounding_box_images_path,
            estimated_body_length=estimated_body_lengths[i],
            number_of_animals=video_params_to_store["number_of_animals"],
            frame_number=global_frame_number,
            pixels=pixels[i],
            pixels_path=pixels_path,
            in_frame_index=i,
            video_height=video_params_to_store["height"],
            video_width=video_params_to_store["width"],
            video_path=video_path,
            frame_number_in_video_path=frame_number_in_video_path,
            resolution_reduction=segmentation_parameters[
                "resolution_reduction"
            ],
        )
        blobs_in_frame.append(blob)

    return blobs_in_frame


def _segment_episode(
    episode_number,
    local_start,
    local_end,
    global_start,
    global_end,
    video_path,
    segmentation_parameters,
    segmentation_data_folder,
    video_params_to_store,
    save_pixels=None,
    save_segmentation_image=None,
):
    """Gets list of blobs segmented in every frame of the episode of the video
    given by `path` (if the video is splitted in different files) or by
    `episode_start_end_frames` (if the video is given in a single file)

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    segmentation_thresholds : dict
        Dictionary with the thresholds used for the segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`
    path : string
        Path to the video file from where to get the VideoCapture (OpenCV) object
    episode_start_end_frames : tuple
        Tuple (starting_frame, ending_frame) indicanting the start and end of the episode
        when the video is given in a single file

    Returns
    -------
    blobs_in_episode : list
        List of `blobs_in_frame` of the episode of the video being segmented
    max_number_of_blobs : int
        Maximum number of blobs found in the episode of the video being segmented

    See Also
    --------
    Video
    Blob
    _get_videoCapture
    segment_frame
    blob_extractor
    """
    # Set file path to store blobs segmentation image and blobs pixels
    if save_segmentation_image == "DISK":
        bounding_box_images_path = os.path.join(
            segmentation_data_folder,
            f"episode_images_{episode_number}.hdf5",
        )
        if os.path.isfile(bounding_box_images_path):
            os.remove(bounding_box_images_path)
    else:
        bounding_box_images_path = None
    if save_pixels == "DISK":
        pixels_path = os.path.join(
            segmentation_data_folder,
            f"episode_pixels_{episode_number}.hdf5",
        )
        if os.path.isfile(pixels_path):
            os.remove(pixels_path)
    else:
        pixels_path = None
    # Read video for the episode
    cap = cv2.VideoCapture(video_path)

    # Get the video on the starting position
    cap.set(1, local_start)

    blobs_in_episode = []
    for (frame_number_in_video_path, global_frame_number) in zip(
        range(local_start, local_end), range(global_start, global_end)
    ):

        blobs_in_frame = _get_blobs_in_frame(
            cap,
            video_params_to_store,
            segmentation_parameters,
            global_frame_number,
            frame_number_in_video_path,
            bounding_box_images_path,
            video_path,
            pixels_path,
            save_pixels,
            save_segmentation_image,
        )

        # store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)

    cap.release()
    return blobs_in_episode


def segment(
    segmentation_parameters: Dict[str, any],
    video_params_to_store: Dict[str, any],
    episodes: List[Tuple[int, int, int, int, int]],
    segmentation_data_folder: str,
    video_paths: List[str],
    number_of_frames: int,
) -> Tuple[List[List[Blob]], int]:
    """
    Computes a list of blobs for each frame of the video and the maximum
    number of blobs found in a frame.

    Parameters
    ----------
    video_path
    segmentation_parameters
    video_attributes_to_store_in_each_blob
    episodes_start_end
    segmentation_data_folder
    video_paths

    Returns
    -------

    See Also
    --------
    _segment_video_in_parallel

    """
    # avoid computing with all the cores in very large videos. It fills the RAM.
    logger.info(f"Pixels stored in {conf.SAVE_PIXELS}")
    logger.info(
        f"Segmentation images stored in {conf.SAVE_SEGMENTATION_IMAGE}"
    )
    num_cpus = int(multiprocessing.cpu_count())
    num_jobs = conf.NUMBER_OF_JOBS_FOR_SEGMENTATION
    if num_jobs is None:
        num_jobs = 1
    elif num_jobs < 0:
        num_jobs = num_cpus + 1 + num_jobs

    set_mkl_to_single_thread()

    logger.info(
        f"Segmenting {len(episodes)} episodes in {num_jobs} parallel jobs"
    )

    blobs_in_episodes = Parallel(n_jobs=num_jobs)(
        delayed(_segment_episode)(
            episode_number,
            local_start,
            local_end,
            global_start,
            global_end,
            video_paths[video_path_index],
            segmentation_parameters,
            segmentation_data_folder,
            video_params_to_store,
            conf.SAVE_PIXELS,
            conf.SAVE_SEGMENTATION_IMAGE,
        )
        for episode_number, (
            local_start,
            local_end,
            video_path_index,
            global_start,
            global_end,
        ) in enumerate(episodes)
    )
    set_mkl_to_multi_thread()

    # blobs_in_episodes is a 3 dimensional list with shape
    # (episode, frame in episode, blob in frame)
    # and we want the 2D list blobs_in_video with shape
    # (global frame, blob in frame)

    blobs_in_video = [[]] * number_of_frames
    for blobs_in_episode, episode_info in zip(blobs_in_episodes, episodes):
        global_start, global_end = episode_info[-2:]
        blobs_in_video[global_start:global_end] = blobs_in_episode

    return blobs_in_video
