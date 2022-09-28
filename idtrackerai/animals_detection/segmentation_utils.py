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

from typing import Optional, Tuple, List
import logging
from rich.progress import track
import cv2
import numpy as np
from confapp import conf

logger = logging.getLogger("__main__.segmentation_utils")


"""
The utilities to segment and extract the blob information
"""


def compute_background(
    video_paths,
    original_ROI,
    episodes,
    number_of_frames_for_background=conf.NUMBER_OF_FRAMES_FOR_BACKGROUND,
):
    """
    Computes the background model by sampling frames from the video with a
    period `background_sampling_period` and computing the stat
    `background_subtraction_stat` across the sampled frames.
    If the video comes in a single file it computes the background in parallel
    splitting the video in `parallel_period`.
    This parameter is ignored if the the video comes in multiple files.

    Parameters
    ----------
    video : idtrackerai.video.VideoObject
    background_sampling_period : int
        sampling period to compute the background model
    background_subtraction_stat: str
        statistic to compute over the sampled frames ("mean", "min", or "max)
    parallel_period: int
        video chunk size (in frames) for the parallel computation
    num_jobs_parallel: int
        number of jobs for the parallel computation
    sigma_gaussian_blur: float
        sigma of the gaussian kernel to blur each frame

    Returns
    -------
    bkg : np.ndarray
        Background model

    """
    mask = original_ROI.astype(bool)

    list_of_frames = []
    for episode in episodes:
        start, end, video_idx = episode[:3]
        list_of_frames += [(frame, video_idx) for frame in range(start, end)]

    frames_to_take = np.linspace(
        0, len(list_of_frames) - 1, number_of_frames_for_background, dtype=int
    )

    frames_for_background = [list_of_frames[i] for i in frames_to_take]

    cap = cv2.VideoCapture(video_paths[0])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames_in_disk = np.empty(
        (len(frames_for_background), height, width), np.uint8
    )
    current_video = 0
    for i, (frame_number, video_idx) in enumerate(
        track(frames_for_background, "Computing background")
    ):
        if video_idx != current_video:
            cap.release()
            cap = cv2.VideoCapture(video_paths[video_idx])
            current_video = video_idx
        if frame_number != int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        assert ret
        frames_in_disk[i] = to_gray_scale(frame)
    cap.release()

    averages = np.asarray(
        [get_frame_average_intensity(frame, mask) for frame in frames_in_disk]
    )

    average = np.mean(averages)

    flickering_factor = averages / average
    for i, frame in enumerate(frames_in_disk):
        cv2.convertScaleAbs(frame, frame, alpha=flickering_factor[i])

    backgrounds = {
        "median": (np.median(frames_in_disk, axis=0) / average).astype(
            np.float32
        ),
        "mean": (np.mean(frames_in_disk, axis=0) / average).astype(np.float32),
        "max": (np.max(frames_in_disk, axis=0) / average).astype(np.float32),
        "min": (np.min(frames_in_disk, axis=0) / average).astype(np.float32),
    }

    return average, backgrounds


def gaussian_blur(frame, sigma=None):
    if sigma is not None and sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), sigma)
    return frame


def to_gray_scale(frame):
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def get_frame_average_intensity(frame: np.ndarray, mask: np.ndarray):
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
    assert mask is not None
    avg = np.float32(np.mean(frame, where=mask.astype(bool)))
    if np.isnan(avg):  # happens when mask is False everywhere
        return np.float32(0.0)
    else:
        return avg


def segment_frame(frame, min_threshold, max_threshold, bkg, ROI, useBkg):
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
    if useBkg:
        # only step where frame normalization is important,
        # because the background is normalised
        frame = cv2.absdiff(bkg, frame)
        p99 = np.percentile(frame, 99.95) * 1.001
        frame = np.clip(255 - frame * (255.0 / p99), 0, 255)
        frame_segmented = cv2.inRange(
            frame, min_threshold, max_threshold
        )  # output: 255 in range, else 0
    else:
        p99 = np.percentile(frame, 99.95) * 1.001
        frame_segmented = cv2.inRange(
            np.clip(frame * (255.0 / p99), 0, 255),
            min_threshold,
            max_threshold,
        )  # output: 255 in range, else 0
    frame_segmented_and_masked = cv2.bitwise_and(
        frame_segmented, frame_segmented, mask=ROI
    )  # Applying the mask
    return frame_segmented_and_masked


def _filter_contours_by_area(
    contours: List, min_area: int, max_area: int
) -> List[np.ndarray]:  # (cnt_points, 1, 2)
    """Filters out contours which number of pixels is smaller than `min_area`
    or greater than `max_area`

    Parameters
    ----------
    contours : list
        List of OpenCV contours
    min_area : int
        Minimum number of pixels for a contour to be acceptable
    max_area : int
        Maximum number of pixels for a contours to be acceptable

    Returns
    -------
    good_contours : list
        List of OpenCV contours that fulfill both area thresholds
    """

    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            good_contours.append(contour)
    return good_contours


def _cnt2BoundingBox(cnt, bounding_box):
    """Transforms the coordinates of the contour in the full frame to the
    bounding box of the blob.

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))


    Returns
    -------
    contour_in_bounding_box : nd.array
        Array with the pairs of coordinates of the contour in the bounding box
    """
    return cnt - np.asarray([bounding_box[0][0], bounding_box[0][1]])


def _get_bounding_box(
    cnt: np.ndarray,
    width: int,
    height: int,
) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]:
    """Computes the bounding box of a given contour with an extra margin of
    constants.EXTRA_PIXELS_BBOX pixels.
    The extra margin is given so that the image can be rotated without adding
    artifacts in the borders. The image will be rotated when setting the
    identification image in the crossing detector step.

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video
    width : int
        Width of the video frame
    height : int
        Height of the video frame

    Returns
    -------
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))
    original_diagonal : int
        Diagonal of the original bounding box computed with OpenCv that serves
        as estimate for the body length of the animal.
    """
    # TODO: rethink whether the expansion is really needed
    x, y, w, h = cv2.boundingRect(cnt)
    original_diagonal = int(np.ceil(np.sqrt(w**2 + h**2)))
    n = conf.EXTRA_PIXELS_BBOX
    if x - n > 0:  # We only expand the
        x = x - n
    else:
        x = 0
    if y - n > 0:
        y = y - n
    else:
        y = 0
    if x + w + 2 * n < width:
        w = w + 2 * n
    else:
        w = width - x
    if y + h + 2 * n < height:
        h = h + 2 * n
    else:
        h = height - y
    expanded_bbox = ((x, y), (x + w, y + h))
    return expanded_bbox, original_diagonal


def _getCentroid(cnt):
    """Computes the centroid of the contour

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video

    Returns
    -------
    centroid : tuple
        (x,y) coordinates of the center of mass of the contour.
    """
    M = cv2.moments(cnt)
    x = M["m10"] / M["m00"]
    y = M["m01"] / M["m00"]
    return (x, y)


def _get_pixels(cnt: np.ndarray, width: int, height: int) -> np.ndarray:
    """Gets the coordinates list of the pixels inside the contour

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in a give
        width and height (it can either be the video width and heigh or the
        bounding box width and height)
    width : int
        Width of the frame
    height : int
        Height of the frame

    Returns
    -------
    pixels_coordinates_list : list
        List of the coordinates of the pixels in a given width and height
    """
    cimg = np.zeros((height, width))
    cv2.drawContours(cimg, [cnt], -1, color=255, thickness=-1)
    pts = np.where(cimg == 255)
    return np.asarray(list(zip(pts[0], pts[1])))


def _get_bounding_box_image(
    frame: np.ndarray,
    cnt: np.ndarray,
    save_pixels: str,
    save_segmentation_image: str,
):
    """Computes the `bounding_box_image`from a given frame and contour. It also
    returns the coordinates of the `bounding_box`, the ravelled `pixels`
    inside of the contour and the diagonal of the `bounding_box` as
    an `estimated_body_length`

    Parameters
    ----------
    frame : nd.array
        frame from where to extract the `bounding_box_image`
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video

    Returns
    -------
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))
    bounding_box_image : nd.array
        Part of the `frame` defined by the coordinates in `bounding_box`
    pixels_in_full_frame_ravelled : list
        List of ravelled pixels coordinates inside of the given contour
    estimated_body_length : int
        Estimated length of the contour in pixels.

    See Also
    --------
    _get_bounding_box
    _cnt2BoundingBox
    _get_pixels
    """
    height = frame.shape[0]
    width = frame.shape[1]
    # Coordinates of an expanded bounding box
    bounding_box, estimated_body_length = _get_bounding_box(
        cnt, width, height
    )  # the estimated body length is the diagonal of the original bounding_box
    # Get bounding box from frame
    if save_segmentation_image == "RAM" or save_segmentation_image == "DISK":
        bounding_box_image = frame[
            bounding_box[0][1] : bounding_box[1][1],
            bounding_box[0][0] : bounding_box[1][0],
        ]
    elif save_segmentation_image == "NONE":
        bounding_box_image = None
    else:
        raise ValueError(
            f"Invalid `save_segmentation_image` = {save_segmentation_image}"
        )
    contour_in_bounding_box = _cnt2BoundingBox(cnt, bounding_box)
    if save_pixels == "RAM" or save_pixels == "DISK":
        pixels_in_bounding_box = _get_pixels(
            contour_in_bounding_box,
            np.abs(bounding_box[0][0] - bounding_box[1][0]),
            np.abs(bounding_box[0][1] - bounding_box[1][1]),
        )
        pixels_in_full_frame = pixels_in_bounding_box + np.asarray(
            [bounding_box[0][1], bounding_box[0][0]]
        )
        pixels_in_full_frame_ravelled = np.ravel_multi_index(
            [pixels_in_full_frame[:, 0], pixels_in_full_frame[:, 1]],
            (height, width),
        )
    elif save_pixels == "NONE":
        pixels_in_full_frame_ravelled = None
    else:
        raise

    return (
        bounding_box,
        bounding_box_image,
        pixels_in_full_frame_ravelled,
        estimated_body_length,
    )


def _get_blobs_information_per_frame(
    frame: np.ndarray,
    contours: List[np.ndarray],
    save_pixels: str,
    save_segmentation_image: str,
):
    """Computes a set of properties for all the `contours` in a given frame.

    Parameters
    ----------
    frame : nd.array
        Frame from where to extract the `bounding_box_image` of every contour
    contours : list
        List of OpenCV contours for which to compute the set of properties

    Returns
    -------
    bounding_boxes : list
        List with the `bounding_box` for every contour in `contours`
    bounding_box_images : list
        List with the `bounding_box_image` for every contour in `contours`
    centroids : list
        List with the `centroid` for every contour in `contours`
    areas : list
        List with the `area` in pixels for every contour in `contours`
    pixels : list
        List with the `pixels` for every contour in `contours`
    estimated_body_lengths : list
        List with the `estimated_body_length` for every contour in `contours`

    See Also
    --------
    _get_bounding_box_image
    _getCentroid
    _get_pixels
    """
    bounding_boxes = []
    bounding_box_images = []
    centroids = []
    areas = []
    pixels = []
    estimated_body_lengths = []

    for i, cnt in enumerate(contours):
        (
            bounding_box,
            bounding_box_image,
            pixels_in_full_frame_ravelled,
            estimated_body_length,
        ) = _get_bounding_box_image(
            frame, cnt, save_pixels, save_segmentation_image
        )
        # bounding boxes
        bounding_boxes.append(bounding_box)
        # bounding_box_images
        bounding_box_images.append(bounding_box_image)
        # centroids
        centroids.append(_getCentroid(cnt))
        areas.append(cv2.contourArea(cnt))
        # pixels lists
        pixels.append(pixels_in_full_frame_ravelled)
        # estimated body lengths list
        estimated_body_lengths.append(estimated_body_length)

    return (
        bounding_boxes,
        bounding_box_images,
        centroids,
        areas,
        pixels,
        estimated_body_lengths,
    )


def blob_extractor(
    segmented_frame: np.ndarray,
    frame: np.ndarray,
    min_area: int,
    max_area: int,
    save_pixels: Optional[str] = "DISK",
    save_segmentation_image: Optional[str] = "DISK",
) -> Tuple[
    List[Tuple],
    List[np.ndarray],
    List[Tuple],
    List[int],
    List[List],
    List[List],
    List[float],
]:
    # TODO: Document
    _, contours, hierarchy = cv2.findContours(
        segmented_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Filter contours by size
    good_contours_in_full_frame = _filter_contours_by_area(
        contours, min_area, max_area
    )
    # get contours properties
    (
        bounding_boxes,
        bounding_box_images,
        centroids,
        areas,
        pixels,
        estimated_body_lengths,
    ) = _get_blobs_information_per_frame(
        frame,
        good_contours_in_full_frame,
        save_pixels,
        save_segmentation_image,
    )

    return (
        bounding_boxes,
        bounding_box_images,
        centroids,
        areas,
        pixels,
        good_contours_in_full_frame,
        estimated_body_lengths,
    )
