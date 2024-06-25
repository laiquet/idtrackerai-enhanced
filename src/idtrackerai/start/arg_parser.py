import ast
from argparse import ArgumentParser, ArgumentTypeError, _ArgumentGroup
from pathlib import Path
from typing import Callable

from idtrackerai import Session
from idtrackerai.utils import conf, resolve_path


def Bool(value: str) -> bool:
    valid = {"true": True, "t": True, "1": True, "false": False, "f": False, "0": False}

    lower_value = value.lower()
    if lower_value not in valid:
        raise ValueError
    return valid[lower_value]


def path(value: str) -> Path:
    return_path = resolve_path(value)
    if not return_path.exists():
        raise ArgumentTypeError(f"No such file or directory: {return_path}")
    return return_path


def pair_of_ints(value: str):
    out = ast.literal_eval(value)
    if not isinstance(out, (tuple, list)):
        raise ValueError

    out = list(out)
    if len(out) != 2:
        raise ValueError
    if any(not isinstance(x, int) for x in out):
        raise ValueError
    return out


def get_parser(defaults: dict | None = None) -> ArgumentParser:
    defaults = defaults or {}

    parser = ArgumentParser(
        prog="idtracker.ai",
        epilog="For more info visit https://idtracker.ai",
        exit_on_error=False,
    )
    parsers: dict[str, ArgumentParser | _ArgumentGroup] = {"base": parser}

    def add_argument(
        name: str, help: str, type: Callable, group: str = "General", **kwargs
    ) -> None:
        name = name.lower()

        metavar = f"<{type.__name__.lower()}>"

        if "choices" in kwargs:
            help += f' (choices: {", ".join(kwargs["choices"])})'

        if name in ("load", "session") or "(default: " in help:
            # Video has a load method, it's not the default for --load
            # name has an adaptative default value
            pass
        elif name.upper() in defaults:
            help += f" (default: {defaults[name.upper()]})"
        elif name.lower() in defaults:
            help += f" (default: {defaults[name.lower()]})"

        if group not in parsers:
            parsers[group] = parsers["base"].add_argument_group(group)

        parsers[group].add_argument(
            "--" + name, help=help + ".", type=type, metavar=metavar, **kwargs
        )

    # General
    add_argument(
        "load",
        help=(
            "A list of .toml files to load session parameters in increasing priority"
            " order"
        ),
        type=path,
        nargs="*",
        dest="parameters",
    )
    parsers["General"].add_argument(
        "--track", help="Track the video without launching the GUI", action="store_true"
    )
    add_argument(
        "name", help="Name of the session (default: name of the video files)", type=str
    )
    add_argument(
        "video_paths",
        help="List of paths to the video files to track",
        type=path,
        nargs="+",
    )
    add_argument(
        "intensity_ths",
        help=(
            "Blob's intensity thresholds. When using background subtraction, the"
            " background difference threshold is the second value of these intensity"
            " thresholds"
        ),
        type=float,
        nargs=2,
    )
    add_argument("area_ths", help="Blob's areas thresholds", type=float, nargs=2)
    add_argument(
        "tracking_intervals",
        help=(
            "Tracking intervals in frames. "
            'Examples: "0,100", "[0,100]", "[0,100] [150,200] ...". '
            "If none, the whole video is tracked"
        ),
        type=pair_of_ints,
        nargs="+",
    )
    add_argument(
        "number_of_animals",
        help="Number of different animals that appear in the video",
        type=int,
    )
    add_argument(
        "use_bkg",
        help="Compute and extract background to improve blob identification",
        type=Bool,
    )
    add_argument(
        "resolution_reduction", help="Video resolution reduction ratio", type=float
    )
    add_argument(
        "exclusive_rois",
        "Treat each separate ROI as closed identities groups",
        type=Bool,
    )
    add_argument(
        "track_wo_identities",
        "Track the video ignoring identities (without AI)",
        type=Bool,
    )
    add_argument(
        "ROI_list",
        help="List of polygons defining the Region Of Interest",
        type=str,
        nargs="+",
    )

    # Output
    add_argument(
        "output_dir",
        help=(
            "Output directory where session folder will be saved to, default is video"
            " paths parent directory"
        ),
        type=path,
        group="Output",
    )
    add_argument(
        "CONVERT_TRAJECTORIES_TO_CSV_AND_JSON",
        "If true, trajectories files are gonna be copied to .csv and .json files",
        type=Bool,
        group="Output",
    )
    add_argument(
        "bounding_box_images_in_ram",
        "If true, bounding box images, a middle step to generate the identification"
        " images, will be kept in RAM until no longer needed. Else, they are saved in"
        " disk and loaded when needed",
        type=Bool,
        group="Output",
    )
    add_argument(
        "ADD_TIME_COLUMN_TO_CSV",
        "If true, adds a time column (in seconds) to csv trajectory files",
        type=Bool,
        group="Output",
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
        group="Output",
    )

    # Background
    add_argument(
        "BACKGROUND_SUBTRACTION_STAT",
        "Statistical method to compute the background",
        type=str,
        choices=["median", "mean", "max", "min"],
        group="Background Subtraction",
    )
    add_argument(
        "NUMBER_OF_FRAMES_FOR_BACKGROUND",
        "Number of frames used to compute the background",
        type=int,
        group="Background Subtraction",
    )

    # Parallel processing
    add_argument(
        "number_of_parallel_workers",
        "Maximum number of jobs to parallelize segmentation and identification"
        " image creation. A negative value means using the number of CPUs in the"
        " system minus the specified value. Zero means using half of the number of"
        " CPUs in the system (limited to 8). One means no multiprocessing at all",
        type=int,
        group="Parallel processing",
    )
    add_argument(
        "FRAMES_PER_EPISODE",
        "Maximum number of frames for each video episode (used to parallelize some"
        " processes)",
        type=int,
        group="Parallel processing",
    )

    # Knowledge transfer
    add_argument(
        "KNOWLEDGE_TRANSFER_FOLDER",
        "Path to the session to transfer knowledge from",
        type=path,
        group="Knowledge and identity transfer",
    )
    add_argument(
        "identity_transfer",
        help="If true, identities from knowledge transfer folder are transferred",
        type=Bool,
        group="Knowledge and identity transfer",
    )
    add_argument(
        "ID_IMAGE_SIZE",
        "The size of the identification images used in the tracking",
        type=int,
        group="Knowledge and identity transfer",
    )

    # Checks
    add_argument(
        "protocol3_action",
        "Choose what to do when protocol 1 and 2 fail and protocol 3 is going to start",
        type=str,
        choices=["ask", "abort", "continue"],
        group="Checks",
    )
    add_argument(
        "check_segmentation",
        help="Check all frames have less or equal number of blobs than animals",
        type=Bool,
        group="Checks",
    )

    # Advanced hyperparameters
    add_argument(
        "THRESHOLD_EARLY_STOP_ACCUMULATION",
        "Ratio of accumulated images needed to early stopping"
        " the accumulation process",
        type=float,
        group="Advanced hyperparameter",
    )
    add_argument(
        "THRESHOLD_ACCEPTABLE_ACCUMULATION",
        "Minimum ratio of accumulated images that an"
        " accumulation process needs to be accepted as successful",
        type=float,
        group="Advanced hyperparameter",
    )
    add_argument(
        "MAXIMAL_IMAGES_PER_ANIMAL",
        "Maximum number of images per animal that will be"
        " used to train the CNN in each accumulation step",
        type=int,
        group="Advanced hyperparameter",
    )
    add_argument(
        "device",
        help='Device name passed to torch.device() to indicate where machine learning computations will be performed, typically "cpu", "cuda", "cuda:0"... See https://pytorch.org/docs/stable/tensor_attributes.html#torch-device. (default: empty string, automatic device selection).',
        type=str,
        group="Advanced hyperparameter",
    )

    # Deprecated
    add_argument(
        "settings",
        help="Use --load with multiple files instead",
        type=path,
        dest="general_settings",
        group="Deprecated",
    )
    add_argument("session", help='Use "--name"', type=str, group="Deprecated")
    return parser


def get_argparser_help():
    """Used to display argument options in docs

    Returns
    -------
    str
        idtracker.ai argument parser help
    """
    return get_parser(Session.__dict__ | conf.as_dict()).format_help()


def parse_args(defaults: dict | None = None):
    parser = get_parser(defaults or (Session.__dict__ | conf.as_dict()))
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


if __name__ == "__main__":
    print(get_argparser_help())
