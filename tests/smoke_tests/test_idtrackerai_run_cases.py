import copy
import logging
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import toml

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Video
from idtrackerai_start_app.__main__ import load_toml
from idtrackerai_start_app.run_idtrackerai import RunIdTrackerAi

COMPRESSED_VIDEO_PATH_B = files("idtrackerai") / "data" / "test_B.avi"
COMPRESSED_VIDEO_PATH_A = files("idtrackerai") / "data" / "test_A.avi"
COMPRESSED_VIDEO_NUM_FRAMES = 508
COMPRESSED_VIDEO_NUM_FRAMES_2 = 501
COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES = 1009
COMPRESSED_VIDEO_WIDTH = 1160
COMPRESSED_VIDEO_HEIGHT = 938
TEST_PARAMS = Path(__file__).parent / "tests_params"
TEMP_DIR = Path(datetime.now().strftime("idtrackerai_pytest_%Y%m%d_%H%M%S")).resolve()

# File tree for tests that use protocol 2
# Since there are many of them that use protocol 2, we define it as a
# global variable
DEFAULT_PROTOCOL_2_TREE = {
    "preprocessing": [
        "list_of_blobs.pickle",
        "list_of_fragments.pickle",
        "list_of_global_fragments.pickle",
        "list_of_blobs_no_gaps.pickle",
    ],
    "crossings_detector": [
        "supervised_crossing_detector_.checkpoint.pth",
        "supervised_crossing_detector_.checkpoint.pth",
    ],
    "segmentation_data": ["episode_images_0.hdf5", "episode_images_1.hdf5"],
    "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
    "accumulation_0": [
        "list_of_fragments.pickle",
        "model_params.json",
        "supervised_identification_network_.checkpoint.pth",
        "supervised_identification_network_.model.pth",
    ],
    "trajectories": ["trajectories.npy", "trajectories_wo_gaps.npy"],
}

DEFAULT_PROTOCOL_2_NO_TREE = {
    "pretraining": [],
    "accumulation_1": [],
    "accumulation_2": [],
    "accumulation_3": [],
}


def run_idtrackerai(
    test_name: str,
    video_paths: list = [COMPRESSED_VIDEO_PATH_B],
    knowledge_transfer_folder=None,
) -> tuple[dict, bool, Path]:
    """Runs idtrackerai using the terminal mode

    It moves to the `root_folder` and from there executes idtrackerai on the
    video `video_path`. The `root_folder` must contain a file called
    `test.json` with the parameters used to run idtrackerai. Some test can also
    contain a file called `local_settings.py` that indicates the advanced
    parameters to be used when running idtrackerai.

    """
    TEMP_DIR.mkdir(exist_ok=True)

    parameters = load_toml((files("idtrackerai") / "constants.toml"))  # type: ignore
    parameters.update(
        {
            "resolution_reduction": 1,
            "check_segmentation": False,
            "ROI_list": None,
            "use_bkg": False,
            "setup_points": None,
            "track_wo_identities": False,
        }
    )
    parameters.update(toml.load((TEST_PARAMS / (test_name + ".toml")).open()))

    parameters["knowledge_transfer_folder"] = knowledge_transfer_folder
    parameters["video_paths"] = video_paths
    parameters["output_dir"] = TEMP_DIR
    expected_output_path = TEMP_DIR / ("session_" + test_name)
    success_flag = RunIdTrackerAi(copy.deepcopy(parameters)).track_video()
    assert expected_output_path.is_dir()
    return parameters, success_flag, expected_output_path


def assert_input_video_object_consistency(input_arguments, session_folder):
    video = Video.load(session_folder)

    assert video.session_folder.name == "session_" + input_arguments["session"]
    assert video.number_of_animals == input_arguments["number_of_animals"]
    assert video.intensity_ths == input_arguments["intensity_ths"]
    assert video.area_ths == input_arguments["area_ths"]
    assert video.check_segmentation == input_arguments.get("check_segmentation", False)

    if not input_arguments.get("use_bkg", False):
        assert video.bkg_model is None
    assert video.track_wo_identities == input_arguments.get(
        "track_wo_identities", False
    )
    assert video.resolution_reduction == input_arguments.get("resolution_reduction", 1)
    # TODO: assert well tracking interval for single and multiple
    # TODO: assert well apply_roi vs roi.


def assert_files_tree(
    tree: dict[str, list[str]], session_folder: Path, expectation=True
):
    for folder, files in tree.items():
        folder_path = session_folder / folder
        if files:
            for file in files:
                assert (folder_path / file).is_file() is expectation
        else:
            assert folder_path.is_dir() is expectation


def assert_list_of_blobs_consistency(
    input_args,
    session_folder: Path,
    num_frames=COMPRESSED_VIDEO_NUM_FRAMES,
    ignore_no_gaps=False,
):
    if ignore_no_gaps:
        blobs_collections = ["list_of_blobs.pickle"]
    else:
        blobs_collections = ["list_of_blobs.pickle", "list_of_blobs_no_gaps.pickle"]

    for blobs_collection in blobs_collections:
        list_of_blobs_path = session_folder / "preprocessing" / blobs_collection

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
    video_object = Video.load(session_folder)

    bkg_model = video_object.bkg_model
    assert bkg_model is not None
    assert bkg_model.shape == (COMPRESSED_VIDEO_HEIGHT, COMPRESSED_VIDEO_WIDTH)
    # background model is computed from normalized frames (divied by the mean
    # of the frame intensity).
    assert abs(bkg_model.mean() - 1) < 0.01


# Test default run with protocol 2
@pytest.fixture(scope="module")
def default_protocol_2_run():
    return run_idtrackerai("test_default_protocol_2")


def test_default_protocol_2_run(default_protocol_2_run):
    input_arguments, success, session_folder = default_protocol_2_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


def test_accumulation_default_protocol2(default_protocol_2_run):
    _, _, session_folder = default_protocol_2_run
    video_object = Video.load(session_folder)
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert video_object.ratio_accumulated_images > 0.9
    # Check that the accumulation attributes are correct
    assert video_object.accumulation_trial == 0
    assert video_object.accumulation_folder.name == "accumulation_0"
    assert video_object.protocol1_timer.has_finished
    assert video_object.protocol2_timer.has_finished
    assert not video_object.protocol3_pretraining_timer.has_finished
    assert not video_object.protocol3_accumulation_timer.has_finished


# Test resolution reduction with ROI
# Test a tracking session that enters into protocol 3
def test_protocol3():
    input_arguments, success, session_folder = run_idtrackerai("test_protocol3")
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    tree = {
        "preprocessing": [
            "list_of_blobs.pickle",
            "list_of_blobs_no_gaps.pickle",
            "list_of_fragments.pickle",
            "list_of_global_fragments.pickle",
        ],
        "segmentation_data": ["episode_images_0.hdf5", "episode_images_1.hdf5"],
        "crossings_detector": [
            "supervised_crossing_detector_.checkpoint.pth",
            "supervised_crossing_detector_.model.pth",
        ],
        "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
        "pretraining": [],
        "accumulation_0": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
        "trajectories": ["trajectories.npy", "trajectories_wo_gaps.npy"],
    }
    assert_files_tree(tree, session_folder)
    video = Video.load(session_folder)
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert video.ratio_accumulated_images < 0.9
    ratios_accumulated_images = [stat[-1][-1] for stat in video.accumulation_statistics]
    assert video.ratio_accumulated_images == max(ratios_accumulated_images)
    best_accumulation = int(np.nanargmax(ratios_accumulated_images))
    assert video.accumulation_trial == best_accumulation
    assert video.accumulation_folder.name == f"accumulation_{best_accumulation}"

    # assert video.protocol1_time != 0  # TODO: protocol 1 time is not correct
    # assert video.protocol2_time != 0  # TODO: protocol 2 time is not correct
    assert video.protocol3_pretraining_timer.has_finished
    assert video.protocol3_accumulation_timer.has_finished
    assert video.pretraining_folder
    assert video.pretraining_folder.name == "pretraining"


# Test single animal run of idtracker.ai
@pytest.fixture(scope="module")
def single_animal_run():
    return run_idtrackerai("test_single_animal")


def test_single_animal(single_animal_run):
    input_arguments, success, session_folder = single_animal_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, ignore_no_gaps=True
    )
    tree = {
        "preprocessing": ["list_of_blobs.pickle"],
        "crossings_detector": [],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": ["episode_images_0.hdf5"],
        # Here they all appear because they are set in the video_object before
        # creating them # TODO: make this similar to segmentation
        # If no need to analyse frame do not create id_images_{}.hdf5
        "identification_images": ["id_images_0.hdf5"],
        "trajectories": ["trajectories.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {"accumulation_0": [], "trajectories": ["trajectories_wo_gaps"]}
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


# Test no identities feature
@pytest.fixture(scope="module")
def wo_identification_run():
    return run_idtrackerai("test_wo_identification")


def test_wo_identification(wo_identification_run):
    input_arguments, success, session_folder = wo_identification_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, ignore_no_gaps=True
    )
    tree = {
        "preprocessing": ["list_of_blobs.pickle"],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": ["episode_images_0.hdf5", "episode_images_1.hdf5"],
        "crossings_detector": [
            "supervised_crossing_detector_.checkpoint.pth",
            "supervised_crossing_detector_.model.pth",
        ],
        "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
        "trajectories": ["trajectories_wo_identification.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {
        "trajectories": ["trajectories.npy", "trajectories_wo_gaps.npy"],
        "accumulation_0": [],
    }
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_wo_identification_crossing_no_identified(wo_identification_run):
    _, _, session_folder = wo_identification_run
    list_of_blobs_path = session_folder / "preprocessing" / "list_of_blobs.pickle"
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    # Crossing are not assigned an identitiy
    assert all(
        blob.identity is None
        for blobs_in_frame in list_of_blobs.blobs_in_video
        for blob in blobs_in_frame
        if blob.is_a_crossing
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        blob.identity is not None
        for blobs_in_frame in list_of_blobs.blobs_in_video
        for blob in blobs_in_frame
        if blob.is_an_individual
    )


# Test single global fragment
@pytest.fixture(scope="module")
def single_global_fragment_run():
    return run_idtrackerai("test_single_global_fragment")


def test_single_global_fragment(single_global_fragment_run):
    input_arguments, success, session_folder = single_global_fragment_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, ignore_no_gaps=True
    )
    tree = {
        "preprocessing": [
            "list_of_blobs.pickle",
            "list_of_fragments.pickle",
            "list_of_global_fragments.pickle",
        ],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": ["episode_images_0.hdf5"],
        "crossings_detector": [],
        "identification_images": ["id_images_0.hdf5"],
        "trajectories": ["trajectories.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {"trajectories": ["trajectories_wo_gaps.npy"], "accumulation_0": []}
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_single_global_fragment_crossing_no_identified(single_global_fragment_run):
    _, _, session_folder = single_global_fragment_run
    list_of_blobs_path = session_folder / "preprocessing" / "list_of_blobs.pickle"
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    # Crossing are not assigned an identitiy
    assert all(
        blob.identity is None
        for blobs_in_frame in list_of_blobs.blobs_in_video
        for blob in blobs_in_frame
        if blob.is_a_crossing
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        blob.identity is not None
        for blobs_in_frame in list_of_blobs.blobs_in_video
        for blob in blobs_in_frame
        if blob.is_an_individual
    )


def test_single_global_fragment_single_global_fragment(single_global_fragment_run):
    input_arguments, _, session_folder = single_global_fragment_run
    fragments_path = session_folder / "preprocessing" / "list_of_fragments.pickle"
    list_of_fragments = ListOfFragments.load(fragments_path)
    assert list_of_fragments.number_of_fragments == input_arguments["number_of_animals"]

    global_fragments_path = (
        session_folder / "preprocessing" / "list_of_global_fragments.pickle"
    )
    list_of_global_fragments = ListOfGlobalFragments.load(global_fragments_path)
    assert list_of_global_fragments.number_of_global_fragments == 1


# Test a video with more blobs than number of animals where the flag
# _chcksegm is set to False
@pytest.fixture(scope="module")
def more_blobs_than_animals_chcksegm_false_run():
    return run_idtrackerai("test_more_blobs_than_animals_chcksegm_false")


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
    # FIXME sometimes it gets to protocol3, sometimes not
    # assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    # assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


def test_more_blobs_than_animals_chcksegm_false_more_blobs_than_animals(
    more_blobs_than_animals_chcksegm_false_run,
):
    (input_arguments, _, session_folder) = more_blobs_than_animals_chcksegm_false_run
    list_of_blobs_path = session_folder / "preprocessing" / "list_of_blobs.pickle"
    number_of_animals = input_arguments["number_of_animals"]
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    assert any(
        len(blobs_in_frame) > number_of_animals
        for blobs_in_frame in list_of_blobs.blobs_in_video
    )


# TODO: Code more_blobs_than_animals_chcksegm_true


# Forcing background subtraction to use the mean statistic creates
# more blobs than animals in some frames
# Test a segmentation with more blobs than number of animals where the flag
# _chcksegm is set to True
@pytest.fixture(scope="module")
def background_subtraction_mean_run():
    return run_idtrackerai("test_bkg_subtraction_mean")


def test_bkg_subtraction_mean_run(background_subtraction_mean_run):
    (input_arguments, success, session_folder) = background_subtraction_mean_run
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
        "preprocessing": ["list_of_blobs.pickle"],
        "segmentation_data": ["episode_images_0.hdf5", "episode_images_1.hdf5"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {"crossings_detector": [], "trajectories": [], "accumulation_0": []}
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_background_subtraction_mean_bkg_model(background_subtraction_mean_run):
    _, _, session_folder = background_subtraction_mean_run
    assert_background_model(session_folder)


# Test tracking a video using background subtraction
# (default uses median statistic)
@pytest.fixture(scope="module")
def background_subtraction_run():
    return run_idtrackerai("test_bkg_subtraction_default")


def test_background_subtraction_run(background_subtraction_run):
    (input_arguments, success, session_folder) = background_subtraction_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {"accumulation_1": [], "accumulation_2": [], "accumulation_3": []}
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_background_subtraction_default_bkg_model(background_subtraction_run):
    _, _, session_folder = background_subtraction_run
    assert_background_model(session_folder)


# Test ROI with BKG
@pytest.fixture(scope="module")
def background_subtraction_with_ROI_run():
    return run_idtrackerai("test_bkg_roi")


def test_background_subtraction_with_ROI_run(background_subtraction_with_ROI_run):
    (input_arguments, success, session_folder) = background_subtraction_with_ROI_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


def test_background_subtraction_with_ROI_bkg_model(background_subtraction_with_ROI_run):
    _, _, session_folder = background_subtraction_with_ROI_run
    assert_background_model(session_folder)


# Test multiple files
@pytest.fixture(scope="module")
def multiple_files_run():
    return run_idtrackerai(
        "test_multiple_files",
        video_paths=[COMPRESSED_VIDEO_PATH_A, COMPRESSED_VIDEO_PATH_B],
    )


def test_multiple_files_run(multiple_files_run):
    input_arguments, success, session_folder = multiple_files_run
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES,
    )
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


# Test knowledge transfer
def test_knowledge_transfer(default_protocol_2_run, caplog):
    _, _, session_folder = default_protocol_2_run
    caplog.set_level(logging.DEBUG)
    input_arguments, success, session_folder = run_idtrackerai(
        "test_knowledge_transfer",
        video_paths=[COMPRESSED_VIDEO_PATH_A],
        knowledge_transfer_folder=session_folder / "accumulation_0",
    )
    assert "Tracking with knowledge transfer" in caplog.text
    assert "Reinitializing fully connected layers" in caplog.text
    assert success
    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, num_frames=COMPRESSED_VIDEO_NUM_FRAMES_2
    )
    video_object = Video.load(session_folder)
    assert video_object.knowledge_transfer_folder


# Test identity transfer
# This also tests protocol 1
def test_identity_transfer(default_protocol_2_run, caplog):
    _, _, session_folder = default_protocol_2_run
    caplog.set_level(logging.DEBUG)
    input_arguments, success, session_folder = run_idtrackerai(
        "test_identity_transfer",
        video_paths=[COMPRESSED_VIDEO_PATH_A],
        knowledge_transfer_folder=session_folder / "accumulation_0",
    )
    assert success
    assert "Tracking with knowledge transfer" in caplog.text
    assert "Identity transfer. Not reinitializing the fully" in caplog.text
    assert "Identities transferred successfully" in caplog.text
    assert "Transferring identities from " in caplog.text
    assert "Protocol 1 successful" in caplog.text

    assert_input_video_object_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, num_frames=COMPRESSED_VIDEO_NUM_FRAMES_2
    )
    video_object = Video.load(session_folder)
    assert video_object.knowledge_transfer_folder
    assert video_object.identity_transfer
    # TODO: This is not truly a user defined parameter
    assert video_object.id_image_size == [42, 42, 1]


# TODO: Code test max_number_of_blobs < number_of_animals
# TODO: Code test save segmentation images
# TODO: Code test data policy
# TODO: Code test save CSV data
# TODO: Code test lower MAX_RATIO_OF_PRETRAINED_IMAGES
# TODO: Code test sigma blurring

# def pytest_sessionfinish(session, exitstatus):
#     shutil.rmtree(TEMP_DIR)
