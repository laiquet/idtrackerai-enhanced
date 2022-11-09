import pytest
from idtrackerai.animals_detection.segmentation_utils import (
    to_gray_scale,
    get_frame_average_intensity,
    gaussian_blur,
)
import cv2
import numpy as np
from importlib.resources import files


TEST_VIDEO_SHAPE = (938, 1160)
TEST_VIDEO_COMPRESSED_PATH = (
    files("idtrackerai")
    / "data"
    / "example_video_compressed"
    / "conflict3and4_20120316T155032_14_compressed.avi"
)
TEST_VIDEO_COMPRESSED_PATH_2 = (
    files("idtrackerai")
    / "data"
    / "example_video_compressed"
    / "conflict3and4_20120316T155032_13_compressed.avi"
)


def test_data_exists():
    assert TEST_VIDEO_COMPRESSED_PATH.is_file()
    assert TEST_VIDEO_COMPRESSED_PATH_2.is_file()


@pytest.fixture()
def video_frame_0():
    cap = cv2.VideoCapture(str(TEST_VIDEO_COMPRESSED_PATH))
    ret, im = cap.read()
    assert ret
    return im


@pytest.fixture()
def video_frame_0_gray(video_frame_0):
    gray = to_gray_scale(video_frame_0)
    assert gray.ndim == 2
    assert gray.shape == TEST_VIDEO_SHAPE
    return gray


mask_from_roi = np.zeros(TEST_VIDEO_SHAPE, bool)
mask_from_roi[10:900, 10:900] = 1
cases = [
    mask_from_roi,
    np.ones(TEST_VIDEO_SHAPE, bool),  # No mask
    np.zeros(TEST_VIDEO_SHAPE, bool),  # All masked
]


@pytest.mark.parametrize("mask", cases)
def test_get_frame_average_intensity(video_frame_0_gray, mask):
    if np.sum(mask) == 0:
        expected_av_intensity = np.float32(0)
    else:
        expected_av_intensity = np.nanmean(
            video_frame_0_gray[mask == 1]
        ).astype(np.float32)
    av_itensity = get_frame_average_intensity(video_frame_0_gray, mask)

    assert np.dtype(av_itensity) == np.float32
    assert av_itensity >= 0
    assert av_itensity <= 255
    np.testing.assert_almost_equal(expected_av_intensity, av_itensity, 3)


cases = [(None, "same"), (0, "same"), (10, "diff")]


@pytest.mark.parametrize("sigma, expect", cases)
def test_gaussian_blur(video_frame_0_gray, sigma, expect):
    blurred_frame = gaussian_blur(video_frame_0_gray, sigma)
    if expect == "same":
        np.testing.assert_equal(video_frame_0_gray, blurred_frame)
    else:  # expect == "diff"
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_equal,
            video_frame_0_gray,
            blurred_frame,
        )
