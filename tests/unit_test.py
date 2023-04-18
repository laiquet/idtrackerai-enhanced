from importlib.resources import files

import cv2
import numpy as np
import pytest

from idtrackerai.animals_detection.segmentation import gaussian_blur, to_gray_scale

TEST_VIDEO_SHAPE = (938, 1160)
TEST_VIDEO_COMPRESSED_PATH_B = files("idtrackerai") / "data" / "test_B.avi"
TEST_VIDEO_COMPRESSED_PATH_A = files("idtrackerai") / "data" / "test_A.avi"


def test_data_exists():
    assert TEST_VIDEO_COMPRESSED_PATH_B.is_file()
    assert TEST_VIDEO_COMPRESSED_PATH_A.is_file()


@pytest.fixture()
def video_frame_0():
    cap = cv2.VideoCapture(str(TEST_VIDEO_COMPRESSED_PATH_B))
    ret, im = cap.read()
    assert ret
    return im


@pytest.fixture()
def video_frame_0_gray(video_frame_0):
    gray = to_gray_scale(video_frame_0)
    assert gray.ndim == 2
    assert gray.shape == TEST_VIDEO_SHAPE
    return gray


cases = [(None, "same"), (0, "same"), (10, "diff")]


@pytest.mark.parametrize("sigma, expect", cases)
def test_gaussian_blur(video_frame_0_gray, sigma, expect):
    blurred_frame = gaussian_blur(video_frame_0_gray, sigma)
    if expect == "same":
        np.testing.assert_equal(video_frame_0_gray, blurred_frame)
    else:  # expect == "diff"
        np.testing.assert_raises(
            AssertionError, np.testing.assert_equal, video_frame_0_gray, blurred_frame
        )
