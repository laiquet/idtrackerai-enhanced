import os
import pytest
from idtrackerai.animals_detection.segmentation_utils import (
    _get_pixels,
    to_gray_scale,
    get_frame_average_intensity,
    gaussian_blur,
)

import cv2
import idtrackerai.constants as cons

import numpy as np

DATA_FOLDER = os.path.join(cons.IDTRACKERAI_FOLDER, "data")
TEST_VIDEO_COMPRESSED_PATH = os.path.join(
    DATA_FOLDER,
    "example_video_compressed",
    "conflict3and4_20120316T155032_14_compressed.avi",
)
TEST_VIDEO_PROPERTIES = {
    "shape": (938, 1160),
    "width": 1160,
    "height": 938,
    "number_of_channels": 3,
    "frame_rate": 28.07,
    "number_of_frames": 508,
}


@pytest.fixture()
def test_video_cap():
    return cv2.VideoCapture(TEST_VIDEO_COMPRESSED_PATH)


@pytest.fixture()
def test_video_frame_0():
    cap = cv2.VideoCapture(TEST_VIDEO_COMPRESSED_PATH)
    _, frame = cap.read()
    return frame


@pytest.fixture()
def test_video_frame_0_gray():
    cap = cv2.VideoCapture(TEST_VIDEO_COMPRESSED_PATH)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray


def test_to_gray_scale(test_video_frame_0):
    gray = to_gray_scale(test_video_frame_0)
    assert gray.ndim == 2
    assert gray.shape == TEST_VIDEO_PROPERTIES["shape"]


mask_from_roi = np.zeros((TEST_VIDEO_PROPERTIES["shape"]), int)
mask_from_roi[10:900, 10:900] = 1
cases = [
    mask_from_roi,
    np.ones((TEST_VIDEO_PROPERTIES["shape"]), int),  # No mask
    np.zeros((TEST_VIDEO_PROPERTIES["shape"]), int),  # All masked
]


@pytest.mark.parametrize("mask", cases)
def test_get_frame_average_intensity(test_video_frame_0_gray, mask):
    if np.sum(mask) == 0:
        expected_av_intensity = np.float32(0)
    else:
        expected_av_intensity = np.nanmean(
            test_video_frame_0_gray[np.where(mask == 1)]
        ).astype(np.float32)
    av_itensity = get_frame_average_intensity(test_video_frame_0_gray, mask)

    assert np.dtype(av_itensity) == np.float32
    assert av_itensity >= 0
    assert av_itensity <= 255
    np.testing.assert_almost_equal(expected_av_intensity, av_itensity)


cases = [(None, "same"), (0, "same"), (10, "diff")]


@pytest.mark.parametrize("sigma, expect", cases)
def test_gaussian_blur(test_video_frame_0_gray, sigma, expect):
    blurred_frame = gaussian_blur(test_video_frame_0_gray, sigma)
    if expect == "same":
        np.testing.assert_equal(test_video_frame_0_gray, blurred_frame)
    else:  # expect == "diff"
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_equal,
            test_video_frame_0_gray,
            blurred_frame,
        )


def test_get_pixels():
    min_x, max_x = 1, 5
    min_y, max_y = 1, 3
    num_pixels = (max_x - min_x + 1) * (max_y - min_y + 1)
    expected_pixels = np.asarray(
        [
            [i, j]
            for i in range(min_y, max_y + 1)
            for j in range(min_x, max_x + 1)
        ]
    )
    width = 10
    height = 10
    cnt = np.asarray(
        [[[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]]
    )
    pixels = _get_pixels(cnt, width, height)
    assert isinstance(pixels, np.ndarray)
    assert pixels.dtype == np.int64
    assert pixels.shape[0] == num_pixels
    assert pixels.shape[1] == 2
    np.testing.assert_equal(pixels, expected_pixels)
