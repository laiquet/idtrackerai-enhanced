from typing import Tuple, Dict
import json
import numpy as np
import os
from idtrackerai import Video, ListOfBlobs
from idtrackerai_app import main
import tempfile
from distutils.dir_util import copy_tree
import shutil
from datetime import datetime
import pytest
import copy
from pathlib import Path
import logging
from importlib.resources import files

IDTRACKERAI_PATH = files("idtrackerai")
COMPRESSED_VIDEO_PATH = (
    IDTRACKERAI_PATH
    / "data"
    / "example_video_compressed"
    / "conflict3and4_20120316T155032_14_compressed.avi"
)
COMPRESSED_VIDEO_PATH_2 = (
    IDTRACKERAI_PATH
    / "data"
    / "example_video_compressed"
    / "conflict3and4_20120316T155032_13_compressed.avi"
)
COMPRESSED_VIDEO_NUM_FRAMES = 508
COMPRESSED_VIDEO_NUM_FRAMES_2 = 501
COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES = 1009
COMPRESSED_VIDEO_WIDTH = 1160
COMPRESSED_VIDEO_HEIGHT = 938
# Get the path to the folder where all the .json files for the tests are stored
ASSETS_FOLDER = Path(__file__).parent / "tests_params"

# Copy the folder to a temporary folder where data will be stored
TEMP_DIR = Path(
    tempfile.mkdtemp(
        prefix=datetime.now().strftime("idtrackerai_pytest_%Y%m%d_%H%M%S")
    )
)
assert TEMP_DIR.is_dir()
copy_tree(ASSETS_FOLDER, str(TEMP_DIR))

# File tree for tests that use protocol 2
# Since there are many of them that use protocol 2, we define it as a
# global variable
DEFAULT_PROTOCOL_2_TREE = {
    "preprocessing": [
        "blobs_collection.npy",
        "fragments.npy",
        "global_fragments.npy",
        "blobs_collection_no_gaps.npy",
    ],
    "crossings_detector": [
        "supervised_crossing_detector_.checkpoint.pth",
        "supervised_crossing_detector_.checkpoint.pth",
    ],
    "segmentation_data": [
        "episode_images_0.hdf5",
        "episode_pixels_0.hdf5",
        "episode_images_1.hdf5",
        "episode_pixels_1.hdf5",
    ],
    "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
    "accumulation_0": [
        "light_list_of_fragments.npy",
        "model_params.npy",
        "supervised_identification_network_.checkpoint.pth",
        "supervised_identification_network_.model.pth",
    ],
    "trajectories": ["trajectories.npy"],
    "trajectories_wo_gaps": ["trajectories_wo_gaps.npy"],
}

DEFAULT_PROTOCOL_2_NO_TREE = {
    "pretraining": [],
    "accumulation_1": [],
    "accumulation_2": [],
    "accumulation_3": [],
}


def get_video_object(session_folder: Path) -> Video:
    """Load the video object in a given session_folder"""
    return Video.load(session_folder / "video_object.npy")


def run_idtrackerai(
    root_folder: Path, video_paths: list[Path] = [COMPRESSED_VIDEO_PATH]
) -> Tuple[Dict, bool, Path]:
    """Runs idtrackerai using the terminal mode

    It moves to the `root_folder` and from there executes idtrackerai on the
    video `video_path`. The `root_folder` must contain a file called
    `test.json` with the parameters used to run idtrackerai. Some test can also
    contain a file called `local_settings.py` that indicates the advanced
    parameters to be used when running idtrackerai.

    """
    # Change working directory to root_folder to read the local_settings.py
    os.chdir(root_folder)
    json_file_path = root_folder / "test.json"
    assert json_file_path.is_file()

    # Get session name from test.json
    with open("test.json", "r") as f:
        input_arguments = json.load(f)
    session_name = input_arguments["session"]

    # The session folder will be generated next to the video
    original_session_folder = video_paths[0].parent / f"session_{session_name}"

    input_arguments["video_paths"] = video_paths

    # Remove any session folder with the same name from potential previous
    # runs
    if original_session_folder.is_dir():
        shutil.rmtree(original_session_folder)

    assert not original_session_folder.is_dir()
    assert json_file_path.is_file()

    success_flag = main(copy.deepcopy(input_arguments), test=True)

    # We move the session folder that is next to the video in the
    # idtrackerai/data folder to the temporary folder
    moved_session_folder = root_folder / f"session_{session_name}"
    shutil.move(original_session_folder, moved_session_folder)

    return (
        input_arguments,
        success_flag,
        moved_session_folder,
    )


def assert_input_video_object_consistency(input_arguments, session_folder):
    video = get_video_object(session_folder)

    assert video.session_folder.name == "session_" + input_arguments["session"]
    assert video.number_of_animals == input_arguments["number_of_animals"]
    assert video.intensity_ths == input_arguments["intensity_ths"]
    assert video.area_ths == input_arguments["area_ths"]
    assert video.check_segmentation == input_arguments.get(
        "check_segmentation", False
    )

    if not input_arguments.get("use_bkg", False):
        assert video.bkg_model is None
    assert video.track_wo_identities == input_arguments.get(
        "track_wo_identities", False
    )
    assert video.resolution_reduction == input_arguments.get(
        "resolution_reduction", 1
    )
    # TODO: assert well tracking interval for single and multiple
    # TODO: assert well apply_roi vs roi.


def assert_files_tree(
    tree: dict[str, list[str]], session_folder: Path, expectation=True
):
    for folder, files in tree.items():
        folder_path = session_folder / folder
        assert folder_path.is_dir() is expectation
        for file in files:
            assert (folder_path / file).is_file() is expectation


def assert_list_of_blobs_consistency(
    input_args,
    session_folder: Path,
    num_frames=COMPRESSED_VIDEO_NUM_FRAMES,
    ignore_no_gaps=False,
):

    if ignore_no_gaps:
        blobs_collections = ["blobs_collection.npy"]
    else:
        blobs_collections = [
            "blobs_collection.npy",
            "blobs_collection_no_gaps.npy",
        ]

    for blobs_collection in blobs_collections:
        list_of_blobs_path = (
            session_folder / "preprocessing" / blobs_collection
        )

        # if list_of_blobs_path.is_file():  # TODO remove this line
        assert list_of_blobs_path.is_file()
        list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
        assert len(list_of_blobs) == num_frames
        if input_args.get("tracking_intervals", False):
            for start, end in input_args["tracking_intervals"]:
                assert all(list_of_blobs.blobs_in_video[start:end])
        else:
            assert all(list_of_blobs.blobs_in_video)


def assert_background_model(session_folder):
    video_object = get_video_object(session_folder)

    bkg_model = video_object.bkg_model
    assert bkg_model is not None
    assert bkg_model.shape == (
        COMPRESSED_VIDEO_HEIGHT,
        COMPRESSED_VIDEO_WIDTH,
    )
    # background model is computed from normalized frames (divied by the mean
    # of the frame intensity).
    assert abs(bkg_model.mean() - 1) < 0.01


def update_local_settings_with_accumulation_folder(
    root_folder, accumulation_folder
):
    local_settings_path = root_folder / "local_settings.py"
    with open(local_settings_path, "r+") as file:
        content = file.read()
        file.seek(0)
        updated_content = content.replace(
            "path/to/accumulation/folder", str(accumulation_folder)
        )
        file.write(updated_content)
        file.truncate()


# Test default run with protocol 2
@pytest.fixture(scope="module")
def default_protocol_2_run():
    return run_idtrackerai(TEMP_DIR / "test_default_protocol_2")


def test_default_protocol_2_run(default_protocol_2_run):
    input_arguments, success, session_folder = default_protocol_2_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(
        DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False
    )


def test_accumulation_default_protocol2(default_protocol_2_run):
    _, _, session_folder = default_protocol_2_run
    video_object = get_video_object(session_folder)
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert video_object.ratio_accumulated_images > 0.9
    # Check that the accumulation attributes are correct
    assert video_object.accumulation_trial == 0
    assert video_object.accumulation_folder.name == "accumulation_0"
    assert video_object.protocol1_time != 0
    assert video_object.protocol2_time != 0
    assert video_object.protocol3_pretraining_time == 0
    assert video_object.protocol3_accumulation_time == 0


# Test resolution reduction with ROI
# Test a tracking session that enters into protocol 3
def test_protocol3():
    input_arguments, success, session_folder = run_idtrackerai(
        TEMP_DIR / "test_protocol3"
    )
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
            "blobs_collection_no_gaps.npy",
            "fragments.npy",
            "global_fragments.npy",
        ],
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_images_1.hdf5",
            "episode_pixels_0.hdf5",
            "episode_pixels_1.hdf5",
        ],
        "crossings_detector": [
            "supervised_crossing_detector_.checkpoint.pth",
            "supervised_crossing_detector_.model.pth",
        ],
        "identification_images": [
            "id_images_0.hdf5",
            "id_images_1.hdf5",
        ],
        "pretraining": [],
        "accumulation_0": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
        "trajectories": ["trajectories.npy"],
        "trajectories_wo_gaps": ["trajectories_wo_gaps.npy"],
    }
    assert_files_tree(tree, session_folder)
    video = get_video_object(session_folder)
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert video.ratio_accumulated_images < 0.9
    ratios_accumulated_images = [
        stat[-1][-1] for stat in video.accumulation_statistics
    ]
    assert video.ratio_accumulated_images == max(ratios_accumulated_images)
    best_accumulation = int(np.nanargmax(ratios_accumulated_images))
    assert video.accumulation_trial == best_accumulation
    assert (
        video.accumulation_folder.name == f"accumulation_{best_accumulation}"
    )

    # assert video.protocol1_time != 0  # TODO: protocol 1 time is not correct
    # assert video.protocol2_time != 0  # TODO: protocol 2 time is not correct
    assert video.protocol3_pretraining_time != 0
    assert video.protocol3_accumulation_time != 0
    assert video.pretraining_folder
    assert video.pretraining_folder.name == "pretraining"


# Test single animal run of idtracker.ai
@pytest.fixture(scope="module")
def single_animal_run():
    return run_idtrackerai(TEMP_DIR / "test_single_animal")


def test_single_animal(single_animal_run):
    input_arguments, success, session_folder = single_animal_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, ignore_no_gaps=True
    )
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
        ],
        "crossings_detector": [],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_pixels_0.hdf5",
        ],
        # Here they all appear because they are set in the video_object before
        # creating them # TODO: make this similar to segmentation
        # If no need to analyse frame do not create id_images_{}.hdf5
        "identification_images": [
            "id_images_0.hdf5",
        ],
        "trajectories": ["trajectories.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {
        "accumulation_0": [],
        "trajectories_wo_gaps": [],
    }
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


# Test no identities feature
@pytest.fixture(scope="module")
def wo_identification_run():
    return run_idtrackerai(TEMP_DIR / "test_wo_identification")


def test_wo_identification(wo_identification_run):
    input_arguments, success, session_folder = wo_identification_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, ignore_no_gaps=True
    )
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
        ],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_images_1.hdf5",
            "episode_pixels_0.hdf5",
            "episode_pixels_1.hdf5",
        ],
        "crossings_detector": [
            "supervised_crossing_detector_.checkpoint.pth",
            "supervised_crossing_detector_.model.pth",
        ],
        "identification_images": [
            "id_images_0.hdf5",
            "id_images_1.hdf5",
        ],
        "trajectories_wo_identification": [
            "trajectories_wo_identification.npy"
        ],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {
        "trajectories": [],
        "trajectories_wo_gaps": [],
        "accumulation_0": [],
    }
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_wo_identification_crossing_no_identified(wo_identification_run):
    _, _, session_folder = wo_identification_run
    list_of_blobs_path = (
        session_folder / "preprocessing" / "blobs_collection.npy"
    )
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    # Crossing are not assigned an identitiy
    assert all(
        [
            blob.identity is None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_a_crossing
        ]
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        [
            blob.identity is not None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_an_individual
        ]
    )


# Test single global fragment
@pytest.fixture(scope="module")
def single_global_fragment_run():
    return run_idtrackerai(TEMP_DIR / "test_single_global_fragment")


def test_single_global_fragment(single_global_fragment_run):
    input_arguments, success, session_folder = single_global_fragment_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, ignore_no_gaps=True
    )
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
            "fragments.npy",
            "global_fragments.npy",
        ],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_pixels_0.hdf5",
        ],
        "crossings_detector": [],
        "identification_images": [
            "id_images_0.hdf5",
        ],
        "trajectories": ["trajectories.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {
        "trajectories_wo_gaps": [],
        "accumulation_0": [],
    }
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_single_global_fragment_crossing_no_identified(
    single_global_fragment_run,
):
    _, _, session_folder = single_global_fragment_run
    list_of_blobs_path = (
        session_folder / "preprocessing" / "blobs_collection.npy"
    )
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    # Crossing are not assigned an identitiy
    assert all(
        [
            blob.identity is None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_a_crossing
        ]
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        [
            blob.identity is not None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_an_individual
        ]
    )


def test_single_global_fragment_single_global_fragment(
    single_global_fragment_run,
):
    input_arguments, _, session_folder = single_global_fragment_run
    fragments_path = session_folder / "preprocessing" / "fragments.npy"
    list_of_fragments = np.load(fragments_path, allow_pickle=True).item()
    assert len(list_of_fragments) == input_arguments["number_of_animals"]

    global_fragments_path = (
        session_folder / "preprocessing" / "global_fragments.npy"
    )
    list_of_global_fragments = np.load(
        global_fragments_path, allow_pickle=True
    ).item()
    assert list_of_global_fragments.number_of_global_fragments == 1


# Test a video with more blobs than number of animals where the flag
# _chcksegm is set to False
@pytest.fixture(scope="module")
def more_blobs_than_animals_chcksegm_false_run():
    return run_idtrackerai(
        TEMP_DIR / "test_more_blobs_than_animals_chcksegm_false"
    )


def test_more_blobs_than_animals_chcksegm_false_run(
    more_blobs_than_animals_chcksegm_false_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = more_blobs_than_animals_chcksegm_false_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    _, _, session_folder = more_blobs_than_animals_chcksegm_false_run
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(
        DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False
    )


def test_more_blobs_than_animals_chcksegm_false_more_blobs_than_animals(
    more_blobs_than_animals_chcksegm_false_run,
):
    (
        input_arguments,
        _,
        session_folder,
    ) = more_blobs_than_animals_chcksegm_false_run
    list_of_blobs_path = (
        session_folder / "preprocessing" / "blobs_collection.npy"
    )
    number_of_animals = input_arguments["number_of_animals"]
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    assert any(
        [
            len(blobs_in_frame) > number_of_animals
            for blobs_in_frame in list_of_blobs.blobs_in_video
        ]
    )


# TODO: Code more_blobs_than_animals_chcksegm_true

# Forcing background subtraction to use the mean statistic creates
# more blobs than animals in some frames
# Test a segmentation with more blobs than number of animals where the flag
# _chcksegm is set to True
@pytest.fixture(scope="module")
def background_subtraction_mean_run():
    return run_idtrackerai(TEMP_DIR / "test_bkg_subtraction_mean")


def test_bkg_subtraction_mean_run(
    background_subtraction_mean_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = background_subtraction_mean_run
    # Tracking does not return a positive success flag because it is
    # intended to fail when the maximum number of blobs is greater than the
    # number of animals indicated in the input arguments and the chcksegm flag
    # is set to True.
    assert not success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, ignore_no_gaps=True
    )  # ignore_no_gaps because the tracking stops before closing gaps
    assert (session_folder / "inconsistent_frames.csv").exists()

    tree = {
        "preprocessing": ["blobs_collection.npy"],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_pixels_0.hdf5",
            "episode_images_1.hdf5",
            "episode_pixels_1.hdf5",
        ],
        "identification_images": [],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {
        "crossings_detector": [],
        "trajectories": [],
        "trajectories_wo_gaps": [],
        "accumulation_0": [],
    }
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_background_subtraction_mean_bkg_model(
    background_subtraction_mean_run,
):
    _, _, session_folder = background_subtraction_mean_run
    assert_background_model(session_folder)


# Test tracking a video using background subtraction
# (default uses median statistic)
@pytest.fixture(scope="module")
def background_subtraction_run():
    return run_idtrackerai(TEMP_DIR / "test_bkg_subtraction_default")


def test_background_subtraction_run(background_subtraction_run):
    (
        input_arguments,
        success,
        session_folder,
    ) = background_subtraction_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_background_subtraction_default_bkg_model(background_subtraction_run):
    _, _, session_folder = background_subtraction_run
    assert_background_model(session_folder)


# Test ROI with BKG
@pytest.fixture(scope="module")
def background_subtraction_with_ROI_run():
    return run_idtrackerai(TEMP_DIR / "test_bkg_roi")


def test_background_subtraction_with_ROI_run(
    background_subtraction_with_ROI_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = background_subtraction_with_ROI_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(
        DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False
    )


def test_background_subtraction_with_ROI_bkg_model(
    background_subtraction_with_ROI_run,
):
    _, _, session_folder = background_subtraction_with_ROI_run
    assert_background_model(session_folder)


# Test multiple files
@pytest.fixture(scope="module")
def multiple_files_run():
    return run_idtrackerai(
        TEMP_DIR / "test_multiple_files",
        video_paths=[COMPRESSED_VIDEO_PATH, COMPRESSED_VIDEO_PATH_2],
    )


def test_multiple_files_run(
    multiple_files_run,
):
    input_arguments, success, session_folder = multiple_files_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES,
    )
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(
        DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False
    )


# Test knowledge transfer
def test_knowledge_transfer(default_protocol_2_run, caplog):
    _, _, session_folder = default_protocol_2_run
    accumulation_folder = session_folder / "accumulation_0"
    root_folder = TEMP_DIR / "test_knowledge_transfer"
    update_local_settings_with_accumulation_folder(
        root_folder, accumulation_folder
    )
    caplog.set_level(logging.DEBUG)
    input_arguments, success, session_folder = run_idtrackerai(
        root_folder, video_paths=[COMPRESSED_VIDEO_PATH_2]
    )
    assert "Tracking with knowledge transfer" in caplog.text
    assert "Reinitializing fully connected layers" in caplog.text
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_2,
    )
    video_object = get_video_object(session_folder)
    assert video_object.knowledge_transfer_folder


# Test identity transfer
# This also tests protocol 1
def test_identity_transfer(default_protocol_2_run, caplog):
    _, _, session_folder = default_protocol_2_run
    accumulation_folder = session_folder / "accumulation_0"
    root_folder = TEMP_DIR / "test_identity_transfer"
    update_local_settings_with_accumulation_folder(
        root_folder, accumulation_folder
    )

    caplog.set_level(logging.DEBUG)
    input_arguments, success, session_folder = run_idtrackerai(
        root_folder, video_paths=[COMPRESSED_VIDEO_PATH_2]
    )
    assert success
    assert "Tracking with knowledge transfer" in caplog.text
    assert "Identity transfer. Not reinitializing the fully" in caplog.text
    assert "Identities transferred successfully" in caplog.text
    assert "Transferring identities from " in caplog.text
    assert "Protocol 1 successful" in caplog.text

    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_2,
    )
    video_object = get_video_object(session_folder)
    assert video_object.knowledge_transfer_folder
    assert video_object.identity_transfer
    # TODO: This is not truly a user defined parameter
    assert video_object.identification_image_size == (42, 42, 1)


# TODO: Code test max_number_of_blobs < number_of_animals
# TODO: Code test save pixels
# TODO: Code test save segmentation images
# TODO: Code test data policy
# TODO: Code test save CSV data
# TODO: Code test lower MAX_RATIO_OF_PRETRAINED_IMAGES
# TODO: Code test sigma blurring

# def pytest_sessionfinish(session, exitstatus):
#     shutil.rmtree(TEMP_DIR)
