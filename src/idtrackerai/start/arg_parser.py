import ast
from argparse import ArgumentParser, ArgumentTypeError, _ArgumentGroup
from collections.abc import Callable
from pathlib import Path

from idtrackerai import Session, __version__, conf
from idtrackerai.utils import resolve_path


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
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parsers: dict[str, ArgumentParser | _ArgumentGroup] = {"base": parser}

    def add_argument(
        name: str, help: str, type: Callable, group: str = "General", **kwargs
    ) -> None:
        name = name.lower()

        metavar = f"<{type.__name__.lower()}>"

        if "choices" in kwargs:
            help += f' (choices: {", ".join(map(str, kwargs["choices"]))})'

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

        if not help.endswith("."):
            help += "."
        parsers[group].add_argument(
            "--" + name, help=help, type=type, metavar=metavar, **kwargs
        )

    # General
    add_argument(
        "load",
        help=(
            "A list of .toml files to load session parameters in increasing priority order. "
            "Later files override earlier ones for the same parameter."
        ),
        type=path,
        nargs="*",
        dest="parameters",
    )
    parsers["General"].add_argument(
        "--track",
        help="Start tracking directly from the terminal, bypassing the graphical user interface (GUI).",
        action="store_true",
    )
    add_argument(
        "name",
        help="Name of the session. Defaults to the name of the video files if not specified.",
        type=str,
    )
    add_argument(
        "video_paths",
        help="List of paths to the video files to track.",
        type=path,
        nargs="+",
    )
    add_argument(
        "intensity_ths",
        help=(
            "Blob intensity thresholds. If using background subtraction, the second value is the background difference threshold."
        ),
        type=float,
        nargs=2,
    )
    add_argument(
        "area_ths", help="Blob area thresholds (min, max).", type=float, nargs=2
    )
    add_argument(
        "tracking_intervals",
        help=(
            'Tracking intervals in frames. Examples: "0,100", "[0,100]", "[0,100] [150,200] ...". '
            "If not set, the whole video is tracked."
        ),
        type=pair_of_ints,
        nargs="+",
    )
    add_argument(
        "number_of_animals",
        help="Number of different animals that appear in the video.",
        type=int,
    )
    add_argument(
        "use_bkg",
        help="Compute and extract background to improve blob identification.",
        type=Bool,
    )
    add_argument(
        "resolution_reduction",
        help="Video resolution reduction factor for identification images. 0 = maximum reduction, 1 = no reduction.",
        type=float,
    )
    add_argument(
        "exclusive_rois",
        "Treat each separate ROI as a closed group of identities.",
        type=Bool,
    )
    add_argument(
        "track_wo_identities",
        "Track the video without assigning identities (no AI identification).",
        type=Bool,
    )
    add_argument(
        "ROI_list",
        help="List of polygons defining the Region Of Interest (ROI).",
        type=str,
        nargs="+",
    )

    # Output
    add_argument(
        "output_dir",
        help=(
            "Output directory where the session folder will be saved. Defaults to the parent directory of the video files."
        ),
        type=path,
        group="Output",
    )
    add_argument(
        "trajectories_formats",
        "List of formats for saving trajectory files. Choices: h5, npy, csv, pickle.",
        type=str,
        group="Output",
        choices=["h5", "npy", "csv", "pickle"],
        nargs="+",
    )
    add_argument(
        "bounding_box_images_in_ram",
        "If true, bounding box images are kept in RAM until no longer needed. Otherwise, they are saved to disk and loaded as needed.",
        type=Bool,
        group="Output",
    )
    add_argument(
        "DATA_POLICY",
        "Data retention policy for the session folder after successful tracking.",
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
        "Statistical method to compute the background.",
        type=str,
        choices=["median", "mean", "max", "min"],
        group="Background Subtraction",
    )
    add_argument(
        "NUMBER_OF_FRAMES_FOR_BACKGROUND",
        "Number of frames used to compute the background. More frames increase accuracy but also memory and computation time.",
        type=int,
        group="Background Subtraction",
    )

    # Parallel processing
    add_argument(
        "number_of_parallel_workers",
        "Maximum number of parallel workers for segmentation and identification image creation. "
        "Negative: CPUs minus value. Zero: half of CPUs (max 4). One: no multiprocessing.",
        type=int,
        group="Parallel processing",
    )
    add_argument(
        "FRAMES_PER_EPISODE",
        "Maximum number of frames per video chunk (episode) for parallel processing.",
        type=int,
        group="Parallel processing",
    )

    # Knowledge transfer
    add_argument(
        "KNOWLEDGE_TRANSFER_FOLDER",
        "Path to a previous session or accumulation folder to transfer model knowledge from.",
        type=path,
        group="Knowledge and identity transfer",
    )
    add_argument(
        "identity_transfer",
        help="If true, identities from the knowledge transfer folder are transferred to the current session.",
        type=Bool,
        group="Knowledge and identity transfer",
    )
    add_argument(
        "ID_IMAGE_SIZE",
        "Size (in pixels) of the identification images used for tracking.",
        type=int,
        group="Knowledge and identity transfer",
    )

    # Checks
    add_argument(
        "check_segmentation",
        help="Abort tracking if any frame has more blobs than animals (to avoid non-animal blobs contaminating identification).",
        type=Bool,
        group="Checks",
    )

    # Contrastive
    add_argument(
        "DISABLE_CONTRASTIVE",
        "Skip contrastive learning and go directly to the accumulation protocol.",
        type=Bool,
        group="Contrastive",
    )
    add_argument(
        "CONTRASTIVE_MAX_MBYTES",
        "Maximum RAM (in megabytes) for preloading identification images during contrastive training.",
        type=float,
        group="Contrastive",
    )
    add_argument(
        "CONTRASTIVE_MIN_ACCUMULATION",
        "Minimum fraction of images that need to be accumulated for taking the contrastive step as sufficient.",
        type=float,
        group="Contrastive",
    )
    add_argument(
        "CONTRASTIVE_BATCHSIZE",
        "Number of image pairs per batch in contrastive training. Larger values require more GPU memory.",
        type=int,
        group="Contrastive",
    )
    add_argument(
        "CONTRASTIVE_SILHOUETTE_TARGET",
        "Minimum silhouette score (0 to 1) required for contrastive training to finish.",
        type=float,
        group="Contrastive",
    )
    add_argument(
        "contrastive_patience",
        "Number of training steps without improvement before early stopping contrastive training.",
        type=int,
        group="Contrastive",
    )

    # Advanced hyperparameters
    add_argument(
        "THRESHOLD_EARLY_STOP_ACCUMULATION",
        "Fraction of accumulated images needed to early stop the accumulation process.",
        type=float,
        group="Advanced hyperparameter",
    )
    add_argument(
        "MAXIMAL_IMAGES_PER_ANIMAL",
        "Maximum number of images per animal used to train the identification CNN in each accumulation step.",
        type=int,
        group="Advanced hyperparameter",
    )
    add_argument(
        "device",
        help=(
            "Device name for torch.device() (e.g., 'cpu', 'cuda', 'cuda:0', "
            "see https://pytorch.org/docs/stable/tensor_attributes.html#torch-device). "
            "Leave empty for automatic selection."
        ),
        type=str,
        group="Advanced hyperparameter",
    )
    add_argument(
        "torch_compile",
        help="If true, models are compiled with torch.compile for faster execution (may not be compatible "
        "with all devices, see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).",
        type=Bool,
        group="Advanced hyperparameter",
    )

    for deprecated_param in (
        "add_time_column_to_csv",
        "convert_trajectories_to_csv_and_json",
        "protocol3_action",
        "threshold_acceptable_accumulation",
        "maximum_number_of_parachute_accumulations",
        "max_ratio_of_pretrained_images",
    ):
        add_argument(
            deprecated_param,
            help=f"The parameter {deprecated_param!r} has been removed",
            type=str,
            group="Deprecated",
        )
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
