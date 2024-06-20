import copy
import logging
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np
import pytest

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Session
from idtrackerai.base.run import RunIdTrackerAi
from idtrackerai.idmatcherai.main import IdMatcherAi
from idtrackerai.start.__main__ import load_toml
from idtrackerai.utils import IdtrackeraiError, resolve_path
from idtrackerai.video.main import (
    generate_individual_video,
    generate_trajectories_video,
)

TEST_VIDEO_PATHS = {
    "test_A": files("idtrackerai") / "data" / "test_A.avi",
    "test_B": files("idtrackerai") / "data" / "test_B.avi",
}

NUM_FRAMES_VIDEO_B = 508
NUM_FRAMES_VIDEO_A = 501
COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES = 1009
COMPRESSED_VIDEO_WIDTH = 1160
COMPRESSED_VIDEO_HEIGHT = 938
TEST_PARAMS = Path(__file__).parent / "smoke_test_params"
TEMP_DIR = resolve_path(
    "pytest_" + datetime.now().isoformat(timespec="seconds").replace(":", "")
)

# File tree for tests that use protocol 2
# Since there are many of them that use protocol 2, we define it as a
# global variable
DEFAULT_PROTOCOL_2_TREE = {
    "preprocessing": [
        "list_of_blobs.pickle",
        "list_of_fragments.json",
        "list_of_global_fragments.json",
    ],
    "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
    "accumulation_0": [
        "list_of_fragments.json",
        "model_params.json",
        "identification_network.model.pth",
    ],
    "trajectories": ["with_gaps.npy", "without_gaps.npy"],
}

DEFAULT_PROTOCOL_2_NO_TREE = {
    "pretraining": [],
    "accumulation_1": [],
    "accumulation_2": [],
    "accumulation_3": [],
}


def run_idtrackerai(
    test_name: str, knowledge_transfer_folder=None
) -> tuple[dict, bool, Path]:
    """Runs idtrackerai using the terminal mode

    It moves to the `root_folder` and from there executes idtrackerai on the
    video `video_path`. The `root_folder` must contain a file called
    `test.json` with the parameters used to run idtrackerai. Some test can also
    contain a file called `local_settings.py` that indicates the advanced
    parameters to be used when running idtrackerai.

    """

    TEMP_DIR.mkdir(exist_ok=True)

    parameters = {  # defaults for smoke tests
        "roi_list": None,
        "track_wo_identities": False,
        "check_segmentation": False,
        "use_bkg": False,
        "resolution_reduction": 1.0,
        "tracking_intervals": None,
    }
    parameters.update(load_toml(TEST_PARAMS / (test_name + ".toml")))

    parameters["protocol3_action"] = "continue"
    parameters["knowledge_transfer_folder"] = knowledge_transfer_folder
    parameters["video_paths"] = [
        TEST_VIDEO_PATHS[name] for name in parameters["video_paths"]
    ]
    parameters["name"] = test_name.replace("test_", "")
    parameters["output_dir"] = TEMP_DIR
    expected_output_path = TEMP_DIR / ("session_" + parameters["name"])

    session = Session()
    invalid_params = session.set_parameters(**parameters)
    assert not invalid_params
    try:
        success_flag = RunIdTrackerAi(copy.deepcopy(session)).track_video()
    except IdtrackeraiError:
        success_flag = False

    assert expected_output_path.is_dir()
    return parameters, success_flag, expected_output_path


def assert_input_session_consistency(input_arguments, session_folder):
    session = Session.load(session_folder)

    assert session.session_folder.name == "session_" + input_arguments["name"]
    if input_arguments["number_of_animals"] > 0:
        assert session.n_animals == input_arguments["number_of_animals"]
    assert session.intensity_ths == input_arguments["intensity_ths"]
    assert session.area_ths == input_arguments["area_ths"]
    assert session.check_segmentation == input_arguments["check_segmentation"]

    if input_arguments["roi_list"] is not None:
        assert session.ROI_list is not None
        assert session.ROI_mask is not None
    else:
        assert session.ROI_list is None
        assert session.ROI_mask is None

    if not input_arguments["use_bkg"]:
        assert session.bkg_model is None
    assert session.track_wo_identities == input_arguments["track_wo_identities"]
    assert session.resolution_reduction == input_arguments["resolution_reduction"]
    # TODO: assert well tracking interval for single and multiple


def assert_files_tree(
    tree: dict[str, list[str]], session_folder: Path, expectation=True
):
    for folder, tree_files in tree.items():
        folder_path = session_folder / folder
        if tree_files:
            for file in tree_files:
                assert (folder_path / file).is_file() is expectation
        else:
            assert folder_path.is_dir() is expectation


def assert_list_of_blobs_consistency(
    input_args, session_folder: Path, num_frames=NUM_FRAMES_VIDEO_B
):
    blobs_collections = ["list_of_blobs.pickle"]

    for blobs_collection in blobs_collections:
        list_of_blobs_path = session_folder / "preprocessing" / blobs_collection

        # if list_of_blobs_path.is_file():  # TODO remove this line
        assert list_of_blobs_path.is_file()
        list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
        assert len(list_of_blobs) == num_frames
        if input_args["tracking_intervals"]:
            for start, end in input_args["tracking_intervals"]:
                assert all(list_of_blobs.blobs_in_video[start:end])
        else:
            assert all(list_of_blobs.blobs_in_video)


def assert_background_model(session_folder):
    session = Session.load(session_folder)

    bkg_model = session.bkg_model
    assert bkg_model is not None
    assert bkg_model.shape == (COMPRESSED_VIDEO_HEIGHT, COMPRESSED_VIDEO_WIDTH)


@pytest.fixture(scope="module")
def default_video_B():
    return run_idtrackerai("test_default_video_B")


@pytest.fixture(scope="module")
def default_video_A():
    return run_idtrackerai("test_default_video_A")


@pytest.fixture(scope="module")
def single_animal_run():
    return run_idtrackerai("test_single_animal")


@pytest.fixture(scope="module")
def wo_identification_run():
    return run_idtrackerai("test_wo_identification")


@pytest.fixture(scope="module")
def id_img_size():
    return run_idtrackerai("test_id_img_size")


@pytest.fixture(scope="module")
def variable_n_animals_run():
    return run_idtrackerai("test_variable_n_animals")


@pytest.fixture(scope="module")
def exclusive_roi_run():
    return run_idtrackerai("test_exclusive_roi")


@pytest.fixture(scope="module")
def single_global_fragment_run():
    return run_idtrackerai("test_single_global_fragment")


@pytest.fixture(scope="module")
def more_blobs_than_animals_chcksegm_false_run():
    return run_idtrackerai("test_more_blobs_than_animals_chcksegm_false")


@pytest.fixture(scope="module")
def background_subtraction_mean_run():
    return run_idtrackerai("test_bkg_subtraction_mean")


@pytest.fixture(scope="module")
def background_subtraction_run():
    return run_idtrackerai("test_bkg_subtraction_default")


@pytest.fixture(scope="module")
def background_subtraction_with_ROI_run():
    return run_idtrackerai("test_bkg_roi")


@pytest.fixture(scope="module")
def multiple_files_run():
    return run_idtrackerai("test_multiple_files")


def test_default_video_B(default_video_B):
    input_arguments, success, session_folder = default_video_B
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


def test_default_video_A(default_video_A):
    input_arguments, success, session_folder = default_video_A
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, NUM_FRAMES_VIDEO_A
    )
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


def test_session_can_be_loaded_without_video_files(default_video_A):
    _, _, session_folder = default_video_A
    session = Session.load(session_folder)
    original_paths = session.video_paths

    session.video_paths = [Path("/file_does_not_exist")]
    session.save()
    session = Session.load(session_folder)
    assert str(session.video_paths[0].name) == "file_does_not_exist"

    session.video_paths = original_paths
    session.save()


def test_accumulation_default_protocol2(default_video_B):
    _, _, session_folder = default_video_B
    session = Session.load(session_folder)
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert session.ratio_accumulated_images > 0.9
    # Check that the accumulation attributes are correct
    assert session.accumulation_trial == 0
    assert session.accumulation_folder.name == "accumulation_0"
    assert session.timers["Accumulation"].finished
    assert "Protocol 3 pre-training" not in session.timers
    assert "Protocol 3 accumulation" not in session.timers


def test_id_img_size(id_img_size):
    input_arguments, success, session_folder = id_img_size
    assert success
    session = Session.load(session_folder)
    assert session.id_image_size == [45, 45, 1]
    with h5py.File(session.id_images_file_paths[0]) as file:
        assert file["id_images"].shape[-1] == 45  # type: ignore
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


# Test resolution reduction with ROI
# Test a tracking session that enters into protocol 3
def test_protocol3():
    input_arguments, success, session_folder = run_idtrackerai("test_protocol3")
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    tree = {
        "preprocessing": [
            "list_of_blobs.pickle",
            "list_of_fragments.json",
            "list_of_global_fragments.json",
        ],
        "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
        "pretraining": [],
        "accumulation_0": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
        "trajectories": ["with_gaps.npy", "without_gaps.npy"],
    }
    assert_files_tree(tree, session_folder)
    session = Session.load(session_folder)
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert session.ratio_accumulated_images < 0.9
    ratios_accumulated_images = [
        stats["ratio_of_accumulated_images"][-1]
        for stats in session.accumulation_statistics_data
    ]
    assert session.ratio_accumulated_images == max(ratios_accumulated_images)
    best_accumulation = int(np.nanargmax(ratios_accumulated_images))
    assert session.accumulation_trial == best_accumulation
    assert session.accumulation_folder.name == f"accumulation_{best_accumulation}"

    # assert video.protocol2_time != 0  # TODO: protocol 2 time is not correct
    assert session.timers["Protocol 3 pre-training"].finished
    assert session.timers["Protocol 3 accumulation"].finished
    assert session.pretraining_folder
    assert session.pretraining_folder.name == "pretraining"


def test_single_animal(single_animal_run):
    input_arguments, success, session_folder = single_animal_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    tree = {
        "preprocessing": ["list_of_blobs.pickle"],
        # there is a tracking interval so other episodes are not segmented
        "bounding_box_images": ["bbox_images_0.hdf5"],
        "identification_images": ["id_images_0.hdf5"],
        "trajectories": ["with_gaps.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {"accumulation_0": [], "trajectories": ["without_gaps"]}
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_variable_n_animals(variable_n_animals_run):
    input_arguments, success, session_folder = variable_n_animals_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    tree = {
        "preprocessing": ["list_of_blobs.pickle"],
        # there is a tracking interval so other episodes are not segmented
        "bounding_box_images": ["bbox_images_0.hdf5", "bbox_images_1.hdf5"],
        "crossings_detector": ["crossing_detector.model.pth"],
        "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
        "trajectories": ["with_gaps.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {"trajectories": ["without_gaps.npy"], "accumulation_0": []}
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_variable_n_animals_crossing_no_identified(variable_n_animals_run):
    _, _, session_folder = variable_n_animals_run
    list_of_blobs_path = session_folder / "preprocessing" / "list_of_blobs.pickle"
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)

    assert all(
        blob.identity is None for blob in list_of_blobs.all_blobs if blob.is_a_crossing
    )

    assert all(
        blob.identity is not None
        for blob in list_of_blobs.all_blobs
        if blob.is_an_individual
    )


def test_wo_identification(wo_identification_run):
    input_arguments, success, session_folder = wo_identification_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    tree = {
        "preprocessing": ["list_of_blobs.pickle"],
        "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
        "trajectories": ["with_gaps.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {"trajectories": ["without_gaps.npy"], "accumulation_0": []}
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_wo_identification_crossing_no_identified(wo_identification_run):
    _, _, session_folder = wo_identification_run
    list_of_blobs_path = session_folder / "preprocessing" / "list_of_blobs.pickle"
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    # Crossing are not assigned an identitiy
    assert all(
        blob.identity is None for blob in list_of_blobs.all_blobs if blob.is_a_crossing
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        blob.identity is not None
        for blob in list_of_blobs.all_blobs
        if blob.is_an_individual
    )


def test_exclusive_roi(exclusive_roi_run):
    input_arguments, success, session_folder = exclusive_roi_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, num_frames=NUM_FRAMES_VIDEO_A
    )
    session = Session.load(session_folder)

    assert len(session.identities_groups) == 2
    assert len(session.identities_groups["Region_1"]) == 7
    assert len(session.identities_groups["Region_0"]) == 1

    fragments = ListOfFragments.load(session.fragments_path, reconnect=False)

    for frag in fragments.individual_fragments:
        if frag.identity == 0:
            continue  # non identified fragment
        assert (
            frag.identity in session.identities_groups[f"Region_{frag.exclusive_roi}"]
        )


def test_single_global_fragment(single_global_fragment_run):
    input_arguments, success, session_folder = single_global_fragment_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    tree = {
        "preprocessing": [
            "list_of_blobs.pickle",
            "list_of_fragments.json",
            "list_of_global_fragments.json",
        ],
        # there is a tracking interval so other episodes are not segmented
        "identification_images": ["id_images_0.hdf5"],
        "trajectories": ["with_gaps.npy"],
    }
    assert_files_tree(tree, session_folder)
    no_tree = {
        "trajectories": ["without_gaps.npy"],
        "accumulation_0": [],
        "crossings_detector": [],
    }
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_single_global_fragment_crossing_no_identified(single_global_fragment_run):
    _, _, session_folder = single_global_fragment_run
    list_of_blobs_path = session_folder / "preprocessing" / "list_of_blobs.pickle"
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    # Crossing are not assigned an identitiy
    assert all(
        blob.identity is None for blob in list_of_blobs.all_blobs if blob.is_a_crossing
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        blob.identity is not None
        for blob in list_of_blobs.all_blobs
        if blob.is_an_individual
    )


def test_single_global_fragment_single_global_fragment(single_global_fragment_run):
    input_arguments, _, session_folder = single_global_fragment_run
    fragments_path = session_folder / "preprocessing" / "list_of_fragments.json"
    list_of_fragments = ListOfFragments.load(fragments_path)
    assert list_of_fragments.number_of_fragments == input_arguments["number_of_animals"]

    global_fragments_path = (
        session_folder / "preprocessing" / "list_of_global_fragments.json"
    )
    list_of_global_fragments = ListOfGlobalFragments.load(global_fragments_path)
    assert list_of_global_fragments.single_global_fragment


def test_more_blobs_than_animals_chcksegm_false_run(
    more_blobs_than_animals_chcksegm_false_run,
):
    (input_arguments, success, session_folder) = (
        more_blobs_than_animals_chcksegm_false_run
    )
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    _, _, session_folder = more_blobs_than_animals_chcksegm_false_run
    # FIXME sometimes it gets to protocol3, sometimes not
    # assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    # assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


def test_more_blobs_than_animals_chcksegm_false_more_blobs_than_animals(
    more_blobs_than_animals_chcksegm_false_run,
):
    input_arguments, _, session_folder = more_blobs_than_animals_chcksegm_false_run
    list_of_blobs_path = session_folder / "preprocessing" / "list_of_blobs.pickle"
    number_of_animals = input_arguments["number_of_animals"]
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    assert any(
        len(blobs_in_frame) > number_of_animals
        for blobs_in_frame in list_of_blobs.blobs_in_video
    )


# TODO: Code more_blobs_than_animals_chcksegm_true


def test_background_subtraction_mean_run(background_subtraction_mean_run):
    input_arguments, success, session_folder = background_subtraction_mean_run
    # Tracking does not return a positive success flag because it is
    # intended to fail when the maximum number of blobs is greater than the
    # number of animals indicated in the input arguments and the chcksegm flag
    # is set to True.
    assert not success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert (session_folder / "inconsistent_frames.csv").exists()

    tree = {"preprocessing": ["list_of_blobs.pickle"]}
    assert_files_tree(tree, session_folder)
    no_tree = {"crossings_detector": [], "trajectories": [], "accumulation_0": []}
    no_tree.update(DEFAULT_PROTOCOL_2_NO_TREE)
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_background_subtraction_mean_bkg_model(background_subtraction_mean_run):
    _, _, session_folder = background_subtraction_mean_run
    assert_background_model(session_folder)


def test_background_subtraction_run(background_subtraction_run):
    input_arguments, success, session_folder = background_subtraction_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {"accumulation_1": [], "accumulation_2": [], "accumulation_3": []}
    assert_files_tree(no_tree, session_folder, expectation=False)


def test_background_subtraction_default_bkg_model(background_subtraction_run):
    _, _, session_folder = background_subtraction_run
    assert_background_model(session_folder)


def test_background_subtraction_with_ROI_run(background_subtraction_with_ROI_run):
    input_arguments, success, session_folder = background_subtraction_with_ROI_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(input_arguments, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


def test_background_subtraction_with_ROI_bkg_model(background_subtraction_with_ROI_run):
    _, _, session_folder = background_subtraction_with_ROI_run
    assert_background_model(session_folder)


def test_multiple_files_run(multiple_files_run):
    input_arguments, success, session_folder = multiple_files_run
    assert success
    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES,
    )
    assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    assert_files_tree(DEFAULT_PROTOCOL_2_NO_TREE, session_folder, expectation=False)


# Test knowledge transfer
def test_knowledge_transfer(id_img_size, caplog):
    _, _, session_folder = id_img_size
    caplog.set_level(logging.DEBUG)
    input_arguments, success, session_folder = run_idtrackerai(
        "test_knowledge_transfer", knowledge_transfer_folder=session_folder
    )
    assert "Tracking with knowledge transfer" in caplog.text
    assert "Reinitializing only fully connected layers" in caplog.text
    assert success

    session = Session.load(session_folder)

    assert session.id_image_size == [45, 45, 1]
    with h5py.File(session.id_images_file_paths[0]) as file:
        assert file["id_images"].shape[-1] == 45  # type: ignore

    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, num_frames=NUM_FRAMES_VIDEO_A
    )

    assert session.knowledge_transfer_folder


def test_identity_transfer(id_img_size, caplog):
    _, _, session_folder = id_img_size
    caplog.set_level(logging.DEBUG)
    input_arguments, success, session_folder = run_idtrackerai(
        "test_identity_transfer",
        knowledge_transfer_folder=session_folder / "accumulation_0",
    )
    assert success
    session = Session.load(session_folder)
    assert "Identity transfer. Not reinitializing the fully" in caplog.text
    assert session.identity_transfer
    assert session.identity_transfer_succeded
    assert session.knowledge_transfer_folder

    assert session.id_image_size == [45, 45, 1]
    with h5py.File(session.id_images_file_paths[0]) as file:
        assert file["id_images"].shape[-1] == 45  # type: ignore

    assert_input_session_consistency(input_arguments, session_folder)
    assert_list_of_blobs_consistency(
        input_arguments, session_folder, num_frames=NUM_FRAMES_VIDEO_A
    )


def test_idmatcherai(default_video_A, default_video_B):
    _, _, session_A_path = default_video_A
    _, _, session_B_path = default_video_B
    IdMatcherAi([session_A_path, session_B_path])
    tree = {
        "matching_results/session_default_video_A": ["assignments.csv"],
        "matching_results/session_default_video_A/csv": [
            "direct_matches.csv",
            "indirect_matches.csv",
            "joined_matches.csv",
        ],
        "matching_results/session_default_video_A/png": [
            "direct_matches.png",
            "indirect_matches.png",
            "joined_matches.png",
        ],
    }
    assert_files_tree(tree, session_B_path)
    results_path = session_B_path / "matching_results" / "session_default_video_A"
    csv_path = results_path / "csv"
    assert np.loadtxt(csv_path / "direct_matches.csv", delimiter=",").sum() > 100
    assert np.loadtxt(csv_path / "indirect_matches.csv", delimiter=",").sum() > 100
    assert np.loadtxt(csv_path / "joined_matches.csv", delimiter=",").sum() > 100

    assignment = np.loadtxt(
        results_path / "assignments.csv",
        delimiter=",",
        skiprows=1,
        usecols=[0, 1],
        dtype=int,
    )

    expected_assignment = np.array(
        [[1, 1], [2, 3], [3, 8], [4, 2], [5, 7], [6, 5], [7, 4], [8, 6]]
    )
    np.testing.assert_array_equal(assignment, expected_assignment)


def test_video_generator(default_video_A):
    _, _, session_path = default_video_A

    session = Session.load(session_path)
    trajectories: np.ndarray = np.load(
        session.trajectories_folder / "with_gaps.npy", allow_pickle=True
    ).item()["trajectories"]

    generate_individual_video(
        session, trajectories, draw_in_gray=True, starting_frame=80, ending_frame=130
    )

    generate_trajectories_video(
        session,
        trajectories,
        draw_in_gray=True,
        centroid_trace_length=10,
        starting_frame=10,
        ending_frame=80,
    )

    generate_individual_video(
        session,
        trajectories,
        draw_in_gray=False,
        starting_frame=80,
        ending_frame=130,
        miniframe_size=100,
    )

    generate_trajectories_video(
        session,
        trajectories,
        draw_in_gray=False,
        centroid_trace_length=10,
        starting_frame=10,
        ending_frame=80,
    )

    generate_trajectories_video(
        session,
        trajectories,
        draw_in_gray=False,
        centroid_trace_length=10,
        starting_frame=10,
        ending_frame=80,
        no_labels=True,
    )
    tree = {
        "individual_videos": [
            "general.avi",
            "individual_1.avi",
            "individual_2.avi",
            "individual_3.avi",
            "individual_4.avi",
            "individual_5.avi",
            "individual_6.avi",
            "individual_7.avi",
            "individual_8.avi",
        ],
        ".": ["test_A_tracked.avi"],
    }
    assert_files_tree(tree, session_path)


# TODO: Code test max_number_of_blobs < number_of_animals
# TODO: Code test save segmentation images
# TODO: Code test data policy
# TODO: Code test save CSV data
# TODO: Code test lower MAX_RATIO_OF_PRETRAINED_IMAGES
