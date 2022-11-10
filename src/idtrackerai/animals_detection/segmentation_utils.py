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
from rich.progress import track
import cv2
import numpy as np
from idtrackerai.utils import conf, Episode


"""
The utilities to segment and extract the blob information
"""


def generate_frame_stack(
    video_paths,
    episodes: list[Episode],
    n_frames_for_background=None,
    progress_bar=None,
    abort=lambda: False,
):
    if n_frames_for_background is None:
        n_frames_for_background = conf.NUMBER_OF_FRAMES_FOR_BACKGROUND
    logging.info(
        f"Generating frame stack for background subtraction with {n_frames_for_background} samples"
    )

    list_of_frames = []
    for e in episodes:
        list_of_frames += [
            (frame, e.video_path_index)
            for frame in range(e.global_start, e.global_end)
        ]

    frames_to_take = np.linspace(
        0, len(list_of_frames) - 1, n_frames_for_background, dtype=int
    )

    frames_to_sample = [list_of_frames[i] for i in frames_to_take]

    cap = cv2.VideoCapture(str(video_paths[0]))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if abort():
        return
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
        assert ret
        frame_stack[i] = to_gray_scale(frame)
        if abort():
            return
        if progress_bar:
            progress_bar.emit(i)
    return frame_stack


def generate_background_from_frame_stack(
    frame_stack,
    ROI_mask,
    stat=None,
    progress_bar=None,
    abort=lambda: False,
):
    if stat is None:
        stat = conf.BACKGROUND_SUBTRACTION_STAT
    logging.info(f"Computing background from a frame stack using '{stat}'")
    averages = np.asarray(
        [get_frame_average_intensity(frame, ROI_mask) for frame in frame_stack]
    )

    average = np.mean(averages)

    flickering_factor = averages / average
    if abort():
        return
    for i, frame in enumerate(frame_stack):
        cv2.convertScaleAbs(frame, frame, alpha=flickering_factor[i])
        if progress_bar:
            progress_bar.emit(i)
    if abort():
        return

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
        return
    return (bkg / average).astype(np.float32)


def compute_background(
    video_paths,
    original_ROI,
    episodes: list[Episode],
    n_frames_for_background=None,
    stat=None,
    progress_bar=None,
):
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

    background = generate_background_from_frame_stack(
        frame_stack, original_ROI, stat
    )

    return background


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

    if mask is None:
        avg = np.mean(frame, dtype=np.float32)
    else:
        avg = np.mean(frame, where=mask, dtype=np.float32)
    if np.isnan(avg):  # happens when mask is False everywhere
        return np.float32(0.0)
    else:
        return avg


def segment_frame(frame, intensity_thresholds, bkg, ROI, useBkg):
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
            frame, *intensity_thresholds
        )  # output: 255 in range, else 0
    else:
        p99 = np.percentile(frame, 99.95) * 1.001
        frame_segmented = cv2.inRange(
            np.clip(frame * (255.0 / p99), 0, 255), *intensity_thresholds
        )  # output: 255 in range, else 0
    # Applying the mask
    if ROI is not None:
        return frame_segmented * ROI
    else:
        return frame_segmented


def _filter_contours_by_area(
    contours: list[np.ndarray], min_area, max_area
) -> list[np.ndarray]:  # (cnt_points, 1, 2)
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
    good_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            good_contours.append(contour)
            good_areas.append(area)
    return good_contours, good_areas


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


def _get_bounding_box_image(
    frame: np.ndarray, cnt: np.ndarray, save_segmentation_image: str, pad: int
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
    if save_segmentation_image == "NONE":
        return None
    elif save_segmentation_image in ("RAM", "DISK"):
        # Coordinates of an expanded bounding box
        frame_w, frame_h = frame.shape
        x0, y0, w, h = cv2.boundingRect(cnt)
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

        # the estimated body length is the diagonal of the original bounding_box
        # Get bounding box from frame
        bbox_image[y0_margin:y1_margin, x0_margin:x1_margin] = frame[
            y0:y1, x0:x1
        ]
        return bbox_image
    else:
        raise ValueError(
            f"Invalid `save_segmentation_image` = {save_segmentation_image}"
        )
