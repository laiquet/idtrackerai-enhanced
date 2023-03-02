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
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable

import cv2
import h5py
import numpy as np
from rich.progress import track

from idtrackerai import Blob
from idtrackerai.utils import Episode, conf, remove_file


def segment_episode(
    inputs: tuple[Episode, list[Path], Any, Any]
) -> tuple[list[list[Blob]], Episode]:
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
    (episode, video_paths, segmentation_parameters, segmentation_data_folder) = inputs
    # Set file path to store blobs segmentation image and blobs pixels
    bbox_images_path = segmentation_data_folder / f"episode_images_{episode.index}.hdf5"
    remove_file(bbox_images_path)

    # Read video for the episode
    video_path = video_paths[episode.video_path_index]
    cap = cv2.VideoCapture(str(video_path))

    # Get the video on the starting position
    cap.set(1, episode.local_start)
    # TODO get ROI_mask and bkg_model resreducted here in order
    # to avoid to do it in every frame

    blobs_in_episode = []
    for frame_number_in_video_path, global_frame_number in zip(
        range(episode.local_start, episode.local_end),
        range(episode.global_start, episode.global_end),
    ):
        ret, frame = cap.read()
        if ret:
            blobs_in_frame = _get_blobs_in_frame(
                frame, segmentation_parameters, global_frame_number, bbox_images_path
            )
        else:
            logging.error(
                "OpenCV could not read frame "
                f"{frame_number_in_video_path} of {video_path}"
            )
            blobs_in_frame = []

        # store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)

    cap.release()
    return blobs_in_episode, episode


def _get_blobs_in_frame(
    frame, segmentation_parameters, global_frame_number, bbox_images_path
) -> list[Blob]:
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

    See Also
    --------
    Video
    Blob
    segment_frame
    blob_extractor
    """

    _, contours, frame = process_frame(frame, **segmentation_parameters)

    bbox_images = [get_bbox_image(frame, cnt) for cnt in contours]

    blobs_in_frame = create_blobs_objects(
        bbox_images, contours, bbox_images_path, global_frame_number
    )

    return blobs_in_frame


def process_frame(
    frame,
    intensity_ths,
    area_ths,
    ROI_mask,
    bkg_model,
    resolution_reduction,
    sigma_blurring=None,
) -> tuple[list[int], list[np.ndarray], np.ndarray]:
    frame = gaussian_blur(frame, sigma=sigma_blurring)
    # avg_brightness = segmentation_parameters["avg_brightness"]

    # Apply resolution reduction
    if resolution_reduction != 1:
        factor = resolution_reduction
        frame = cv2.resize(
            frame,
            None,  # type: ignore
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_AREA,
        )
        if bkg_model is not None:
            bkg_model = cv2.resize(
                bkg_model,
                None,  # type: ignore
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_AREA,
            )
        if ROI_mask is not None:
            ROI_mask = cv2.resize(
                ROI_mask.astype("uint8"),
                None,  # type: ignore
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_AREA,
            ).astype(bool)
    # Convert the frame to gray scale
    gray = to_gray_scale(frame)
    # Normalize frame
    # flickering_factor = avg_brightness / get_frame_average_intensity(
    #     gray, mask
    # )
    # normalized_framed = cv2.convertScaleAbs(gray, alpha=flickering_factor)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            normalized_framed = gray / get_frame_average_intensity(gray, ROI_mask)
        except RuntimeWarning:
            normalized_framed = gray

    # Binarize frame
    segmentedFrame = segment_frame(
        normalized_framed, intensity_ths, bkg_model, ROI_mask
    )

    # Extract blobs info
    contours = cv2.findContours(
        segmentedFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]

    # Filter contours by size
    areas = []
    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_ths[0] <= area <= area_ths[1]:
            good_contours.append(np.squeeze(contour))
            areas.append(area)
    return areas, good_contours, gray


def create_blobs_objects(
    miniframes, contours, bbox_images_path, global_frame_number
) -> list[Blob]:
    with h5py.File(bbox_images_path, "a") as f1:
        for i, miniframe in enumerate(miniframes):
            f1.create_dataset(f"{global_frame_number}-{i}", data=miniframe)

    blobs_in_frame = [
        Blob(
            contour=contour,
            frame_number=global_frame_number,
            bbox_img_id=f"{global_frame_number}-{i}",
        )
        for i, contour in enumerate(contours)
    ]

    return blobs_in_frame


def segment(
    segmentation_parameters: dict,
    episodes: list[Episode],
    bbox_images_path: Path,
    video_paths: list[Path],
    number_of_frames: int,
) -> list[list[Blob]]:
    """
    Computes a list of blobs for each frame of the video and the maximum
    number of blobs found in a frame.

    Parameters
    ----------
    video_path
    segmentation_parameters
    episodes_start_end
    segmentation_data_folder
    video_paths

    Returns
    -------

    See Also
    --------
    _segment_video_in_parallel

    """
    logging.info("Segmenting video")
    # avoid computing with all the cores in very large videos. It fills the RAM.
    num_jobs = conf.number_of_parallel_workers

    segmentation_parameters["sigma_blurring"] = conf.SIGMA_GAUSSIAN_BLURRING

    logging.info(f"Segmenting {len(episodes)} episodes in {num_jobs} parallel jobs")

    inputs = [
        (episode, video_paths, segmentation_parameters, bbox_images_path.parent)
        for episode in episodes
    ]

    blobs_in_video: list[list[Blob]] = [[]] * number_of_frames
    with Pool(min(num_jobs, len(inputs))) as p:
        for blobs_in_episode, episode in track(
            p.imap_unordered(segment_episode, inputs), "Segmenting video", len(inputs)
        ):
            blobs_in_video[episode.global_start : episode.global_end] = blobs_in_episode

    return blobs_in_video


def generate_frame_stack(
    video_paths,
    episodes: list[Episode],
    n_frames_for_background=None,
    progress_bar=None,
    abort: Callable = lambda: False,
) -> np.ndarray | None:
    if n_frames_for_background is None:
        n_frames_for_background = conf.NUMBER_OF_FRAMES_FOR_BACKGROUND
    logging.info(
        f"Generating frame stack for background subtraction with {n_frames_for_background} samples"
    )

    list_of_frames = []
    for e in episodes:
        list_of_frames += [
            (frame, e.video_path_index) for frame in range(e.local_start, e.local_end)
        ]

    frames_to_take = np.linspace(
        0, len(list_of_frames) - 1, n_frames_for_background, dtype=int
    )

    frames_to_sample = [list_of_frames[i] for i in frames_to_take]

    cap = cv2.VideoCapture(str(video_paths[0]))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if abort():
        return None
    frame_stack = np.empty((len(frames_to_sample), height, width), np.uint8)
    current_video = 0
    for i, (frame_number, video_idx) in enumerate(
        track(frames_to_sample, "Computing background")
    ):
        if video_idx != current_video:
            cap.release()
            cap = cv2.VideoCapture(str(video_paths[video_idx]))
            current_video = video_idx
        if frame_number != int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        assert ret, f"{frame_number = }, {video_idx}"
        frame_stack[i] = to_gray_scale(frame)
        if abort():
            return None
        if progress_bar:
            progress_bar.emit(i)
    return frame_stack


def generate_background_from_frame_stack(
    frame_stack, ROI_mask, stat=None, progress_bar=None, abort: Callable = lambda: False
) -> np.ndarray | None:
    if stat is None:
        stat = conf.BACKGROUND_SUBTRACTION_STAT
    logging.info(f"Computing background from a frame stack using '{stat}'")
    averages = np.asarray(
        [get_frame_average_intensity(frame, ROI_mask) for frame in frame_stack]
    )

    average = np.mean(averages)

    flickering_factor = averages / average
    if abort():
        return None
    for i, frame in enumerate(frame_stack):
        cv2.convertScaleAbs(frame, frame, alpha=flickering_factor[i])
        if progress_bar:
            progress_bar.emit(i)
    if abort():
        return None

    if stat == "median":
        bkg = np.median(frame_stack, axis=0, overwrite_input=True)
    elif stat == "mean":
        bkg = np.mean(frame_stack, axis=0)
    elif stat == "max":
        bkg = np.max(frame_stack, axis=0)
    elif stat == "min":
        bkg = np.min(frame_stack, axis=0)
    else:
        raise ValueError(
            f"Stat '{stat}' is not one of ('median', 'mean', 'max' or 'min')"
        )
    if abort():
        return None
    return (bkg / average).astype(np.float32)


def compute_background(
    video_paths,
    ROI_mask,
    episodes: list[Episode],
    n_frames_for_background=None,
    stat=None,
    progress_bar=None,
) -> np.ndarray | None:
    """
    Computes the background model by sampling `n_frames_for_background` frames
    from the video and computing the stat ('median', 'mean', 'max' or 'min')
    across the sampled frames.

    Parameters
    ----------
    video_paths : list[str]
    original_ROI: np.ndarray
    episodes: list[tuple(int, int, int, int, int)]
    stat: str
        statistic to compute over the sampled frames
        ('median', 'mean', 'max' or 'min')
    sigma_gaussian_blur: float
        sigma of the gaussian kernel to blur each frame

    Returns
    -------
    bkg : np.ndarray
        Background model
    """
    if n_frames_for_background is None:
        n_frames_for_background = conf.NUMBER_OF_FRAMES_FOR_BACKGROUND

    if stat is None:
        stat = conf.BACKGROUND_SUBTRACTION_STAT

    frame_stack = generate_frame_stack(
        video_paths, episodes, n_frames_for_background, progress_bar
    )

    background = generate_background_from_frame_stack(frame_stack, ROI_mask, stat)

    return background


def gaussian_blur(frame: np.ndarray, sigma=None) -> np.ndarray:
    if sigma is not None and sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), sigma)
    return frame


def to_gray_scale(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def get_frame_average_intensity(frame: np.ndarray, mask: np.ndarray) -> np.float32:
    """Computes the average intensity of a given frame considering the maks.
    Only pixels with values
    different than zero in the mask are considered to compute the average
    intensity

    Parameters
    ----------
    frame : nd.array
        Frame from which to compute the average intensity
    mask : nd.array
        Mask to be applied. Pixels with value 0 will be ignored to compute the
        average intensity.

    Returns
    -------

    """

    if mask is None:
        avg = np.mean(frame, dtype=np.float32)
    else:
        avg = np.mean(frame, where=mask, dtype=np.float32)
    # is False everywhere
    return np.float32(0.0) if np.isnan(avg) else avg


def segment_frame(
    frame: np.ndarray,
    intensity_thresholds: list[int],
    bkg: np.ndarray | None,
    ROI: np.ndarray | None,
) -> np.ndarray:
    """Applies the intensity thresholds (`min_threshold` and `max_threshold`)
    and the mask (`ROI`) to a given frame. If `useBkg` is True,
    the background subtraction operation is applied before
    thresholding with the given `bkg`.

    Parameters
    ----------
    frame : nd.array
        Frame to be segmented
    min_threshold : int
        Minimum intensity threshold for the segmentation (value from 0 to 255)
    max_threshold : int
        Maximum intensity threshold for the segmentation (value from 0 to 255)
    bkg : nd.array
        Background model to be used in the background subtraction operation
    ROI : nd.array
        Mask to be applied after thresholding. Ones in the array are pixels to
        be considered, zeros are pixels to be discarded.
    useBkg : bool
        Flag indicating whether background subtraction must be performed or not

    Returns
    -------
    frame_segmented_and_masked : nd.array
        Frame with zeros and ones after applying the thresholding and the mask.
        Pixels with value 1 are valid pixels given the thresholds and the mask.
    """
    if bkg is not None:
        frame = cv2.absdiff(bkg, frame)
        p99 = np.percentile(frame, 99.95) * 1.001
        frame = 255 - np.clip(frame * (255.0 / p99), None, 255)
    else:
        p99 = np.percentile(frame, 99.95) * 1.001
        frame = np.clip(frame * (255.0 / p99), None, 255)

    # TODO why the next lines give errors
    # if useBkg:
    #     frame = cv2.absdiff(bkg, frame)
    #     p99 = np.percentile(frame, 99.95) * 1.001
    #     frame = 255 - cv2.convertScaleAbs(frame, alpha=255 / p99)
    # else:
    #     # TODO optimize next two lines
    #     p99 = np.percentile(frame, 99.95) * 1.001
    #     frame = cv2.convertScaleAbs(frame, alpha=255 / p99)

    frame_segmented = cv2.inRange(frame, *intensity_thresholds)  # type: ignore
    # Applying the mask
    return frame_segmented if ROI is None else frame_segmented * ROI


def get_bbox_image(frame: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Computes the `bbox_image`from a given frame and contour. It also
    returns the coordinates of the `bbox`, the ravelled `pixels`
    inside of the contour and the diagonal of the `bbox` as
    an `estimated_body_length`

    Parameters
    ----------
    frame : nd.array
        frame from where to extract the `bbox_image`
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video

    Returns
    -------
    bbox : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))
    bbox_image : nd.array
        Part of the `frame` defined by the coordinates in `bbox`
    pixels_in_full_frame_ravelled : list
        List of ravelled pixels coordinates inside of the given contour
    estimated_body_length : int
        Estimated length of the contour in pixels.

    See Also
    --------
    _get_bbox
    _cnt2BoundingBox
    _get_pixels
    """
    # extra padding for future image eroding
    pad = 1
    # Coordinates of an expanded bounding box
    frame_w, frame_h = frame.shape
    x0, y0, w, h = cv2.boundingRect(cnt)
    w -= 1  # cv2 adds an extra pixel on width and height
    h -= 1
    x0 -= pad
    y0 -= pad
    x1 = x0 + w + 2 * pad
    y1 = y0 + h + 2 * pad

    if x0 < 0:
        x0_margin = -x0
        x0 = 0
    else:
        x0_margin = 0

    if y0 < 0:
        y0_margin = -y0
        y0 = 0
    else:
        y0_margin = 0

    if x1 > frame_h:
        x1_margin = frame_h - x1
        x1 = frame_h
    else:
        x1_margin = None

    if y1 > frame_w:
        y1_margin = frame_w - y1
        y1 = frame_w
    else:
        y1_margin = None

    bbox_image = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)

    # the estimated body length is the diagonal of the original bbox
    # Get bounding box from frame
    bbox_image[y0_margin:y1_margin, x0_margin:x1_margin] = frame[y0:y1, x0:x1]
    return bbox_image
