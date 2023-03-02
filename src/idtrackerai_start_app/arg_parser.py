import ast
from argparse import ArgumentParser
from pathlib import Path

from idtrackerai.utils import conf


def Bool(value: str):
    valid = {"true": True, "t": True, "1": True, "false": False, "f": False, "0": False}

    lower_value = value.lower()
    if lower_value not in valid:
        raise ValueError
    return valid[lower_value]


def list_of_two_ints(value: str):
    out = ast.literal_eval(value)
    if len(out) != 2:
        raise ValueError
    if any(not isinstance(x, int) for x in out):
        raise ValueError
    return out


def list_of_lists_of_two_ints(value: str):
    out = ast.literal_eval(value)

    if all(isinstance(x, int) for x in out):
        if len(out) != 2:
            raise ValueError
        return out
    elif all(isinstance(x, list) for x in out):
        for sublist in out:
            if any(not isinstance(x, int) for x in sublist) or len(sublist) != 2:
                raise ValueError
        return out
    else:
        raise ValueError


def parse_args():
    parser = ArgumentParser(
        prog="idTracker.ai", epilog="For more info visit https://idtracker.ai"
    )

    def add_argument(name: str, help: str, type, metavar: str = "", **kwargs):
        name = name.lower()

        help = f"({type.__name__.lower()}) {help}"

        if "choices" in kwargs:
            help += f' (choices: {", ".join(kwargs["choices"])})'

        if hasattr(conf, name):
            help += f" (default: {getattr(conf, name)})"

        parser.add_argument(
            "--" + name, help=help, type=type, metavar=metavar, **kwargs
        )

    add_argument(
        "load",
        help="Primary .toml file to load session parameters",
        type=Path,
        metavar="session_parameters",
        dest="session_parameters",
    )

    add_argument(
        "settings",
        help="Secondary .toml file to load general settings",
        type=Path,
        metavar="general_settings",
        dest="general_settings",
    )

    parser.add_argument(
        "--track", help="Track the video without launching the GUI", action="store_true"
    )

    add_argument(
        "tracking_intervals",
        help="Tracking intervals in frames. "
        "Examples: '[0,100]', '[[0,100],[150,200],...]'. "
        "If none, the whole video is tracked",
        type=list_of_lists_of_two_ints,
    )
    add_argument(
        "identity_transfer",
        help="If true, identities from knowledge transfer folder are transferred",
        type=Bool,
    )
    add_argument(
        "intensity_ths", help="Pixel's intensity thresholds", type=list_of_two_ints
    )
    add_argument("area_ths", help="Blob's areas thresholds", type=list_of_two_ints)
    add_argument(
        "number_of_animals",
        help="Number of different animals that appear in the video",
        type=int,
    )
    add_argument(
        "output_dir",
        help="Output directory where session folder will be saved to, "
        "default is video paths parent directory",
        type=Path,
    )
    add_argument(
        "resolution_reduction", help="Video resolution reduction ratio", type=float
    )
    add_argument(
        "check_segmentation",
        help="Check all frames have less or equal number of blobs than animals",
        type=Bool,
    )
    add_argument(
        "ROI_list", help="List of polygons defining the Region Of Interest", type=str
    )
    add_argument(
        "use_bkg",
        help="Compute and extract background to improve blob identification",
        type=Bool,
    )
    add_argument(
        "video_paths",
        help="List of paths to the video files to track",
        type=str,
        nargs="+",
    )
    add_argument("session", help="Name of the session", type=str, default=None)
    add_argument(
        "track_wo_identities",
        "Track the video ignoring identities (without AI)",
        type=Bool,
    )
    add_argument(
        "CONVERT_TRAJECTORIES_TO_CSV_AND_JSON",
        "If true, trajectories files are gonna be copied to .csv and .json files",
        type=Bool,
    )
    add_argument(
        "FRAMES_PER_EPISODE",
        "Maximum number of frames for each video episode "
        "(used to parallelize some processes)",
        type=int,
    )
    add_argument(
        "KNOWLEDGE_TRANSFER_FOLDER",
        "Path to the session to transfer knowledge from",
        type=Path,
    )
    add_argument(
        "BACKGROUND_SUBTRACTION_STAT",
        "Statistical method to compute the background",
        type=str,
        choices=["median", "mean", "max", "min"],
    )
    add_argument(
        "NUMBER_OF_FRAMES_FOR_BACKGROUND",
        "Number of frames used to compute the background",
        type=int,
    )
    add_argument(
        "number_of_parallel_workers",
        "Maximum number of jobs to parallelize segmentation and "
        "identification image creation. A negative value means using the number "
        "of CPUs in the system minus the specified value. Zero means using half "
        "of the number of CPUs in the system",
        type=int,
    )
    add_argument(
        "DATA_POLICY",
        "Type of data policy indicating the data in the session folder not to be"
        "erased when successfully finished a tracking",
        choices=[
            "trajectories",
            "validation",
            "knowledge_transfer",
            "idmatcher.ai",
            "all",
        ],
        type=str,
    )
    add_argument(
        "IDENTIFICATION_IMAGE_SIZE",
        "The size of the identification images used in the tracking",
        type=int,
    )

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    for key, value in args.items():
        if isinstance(value, Path):
            args[key] = value.expanduser().resolve()
    return args
