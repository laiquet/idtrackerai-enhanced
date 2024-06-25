import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
from shutil import rmtree
from typing import IO, Iterable, Sequence, Type, TypeVar

import cv2
import h5py
import numpy as np
import toml
from rich.progress import BarColumn, Progress, TaskProgressColumn, TimeRemainingColumn


class IdtrackeraiError(Exception):
    def __str__(self) -> str:
        # add __cause__ to string representation
        out = super().__str__()
        if self.__cause__ is None:
            return out
        else:
            return (
                f"{out}\n    Original error message: {self.__cause__}"
                if out
                else str(self.__cause__)
            )


InputType = TypeVar("InputType")


def track(
    sequence: Iterable[InputType],
    desc: str = "Working...",
    total: float | None = None,
    verbose: bool = True,
) -> Iterable[InputType]:
    """A custom interpretation of rich.progress.track"""

    if not verbose:
        yield from sequence
        return

    progress = Progress(
        "         " + desc,
        BarColumn(bar_width=None),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True),
        transient=True,
    )

    with progress:
        yield from progress.track(sequence, total, description=desc)

    task = progress.tasks[0]

    logging.info(
        "[green]%s[/] (%s iterations). It took %s",
        desc,
        int(task.total) if task.total is not None else "unknown",
        "--:--" if task.elapsed is None else timedelta(seconds=int(task.elapsed)),
        stacklevel=2,
        extra={"markup": True},
    )


def delete_attributes_from_object(object_to_modify, list_of_attributes):
    for attribute in list_of_attributes:
        if hasattr(object_to_modify, attribute):
            delattr(object_to_modify, attribute)


def load_toml(path: Path, name: str | None = None) -> dict:
    # we do not check if file exists, pathlib will do it for us

    # Avoid loading huge video files loaded by mistake in CLI with "--load"
    if path.stat().st_size > 5000000:
        raise IdtrackeraiError(
            f"{path} takes {path.stat().st_size/(1024**2):.1f} MB, it does not seem"
            " like a .toml file"
        )

    try:
        toml_dict = {
            key.lower(): value for key, value in toml.load(path.open()).items()
        }

        for key, value in toml_dict.items():
            if value == "":
                toml_dict[key] = None

        logging.info(
            pprint_dict(toml_dict, name or f"Loading parameters from {path}"),
            extra={"markup": True},
        )
        return toml_dict
    except Exception as exc:
        raise IdtrackeraiError(f"Could not read toml file {path}") from exc


def create_dir(path: Path, remove_existing=False):
    if path.is_dir():
        if remove_existing:
            rmtree(path)
            path.mkdir()
            logging.info(f"Directory {path} has been emptied", stacklevel=2)
        else:
            logging.info(f"Directory {path} already exists", stacklevel=2)
    else:
        if not path.parent.is_dir():
            path.parent.mkdir()
        path.mkdir()
        logging.info(f"Directory {path} has been created", stacklevel=2)


def remove_dir(path: Path):
    if path.is_dir():
        rmtree(path, ignore_errors=True)
        logging.info(f"Directory {path} has been removed", stacklevel=2)
    else:
        logging.info(f"Directory {path} not found, can't remove", stacklevel=2)


def remove_file(path: Path):
    if path.is_file():
        path.unlink()
        logging.info(f"File {path} has been removed", stacklevel=2)


def assert_all_files_exist(paths: list[Path]):
    """Returns FileNotFoundError if any of the paths is not an existing file"""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"File {path} not found")


def get_vertices_from_label(label: str, close=False) -> np.ndarray:
    """Transforms a string representation of a polygon from the
    ROI widget (idtrackerai_app) into a vertices np.array"""
    try:
        data = json.loads(label[10:].replace("'", '"'))
    except ValueError as exc:
        raise IdtrackeraiError(f'Not recognized ROI representation: "{label}"') from exc

    if label[2:9] == "Polygon":
        vertices = np.asarray(data)
    elif label[2:9] == "Ellipse":
        vertices = np.asarray(
            cv2.ellipse2Poly(data["center"], data["axes"], data["angle"], 0, 360, 2)
        )
    else:
        raise TypeError(label)

    if close:
        return np.vstack([vertices, vertices[0]])
    return vertices


def build_ROI_mask_from_list(
    list_of_ROIs: None | list[str] | str,
    resolution_reduction: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Transforms a list of polygons (as type str) from
    ROI widget (idtrackerai_app) into a boolean np.array mask"""

    if list_of_ROIs is None:
        return None
    ROI_mask = np.zeros(
        (
            int(height * resolution_reduction + 0.5),
            int(width * resolution_reduction + 0.5),
        ),
        np.uint8,
    )

    if isinstance(list_of_ROIs, str):
        list_of_ROIs = [list_of_ROIs]

    for line in list_of_ROIs:
        vertices = (get_vertices_from_label(line) * resolution_reduction + 0.5).astype(
            np.int32
        )
        if line[0] == "+":
            cv2.fillPoly(ROI_mask, (vertices,), color=[255])
        elif line[0] == "-":
            cv2.fillPoly(ROI_mask, (vertices,), color=[0])
        else:
            raise TypeError
    return ROI_mask


@dataclass(slots=True)
class Episode:
    index: int
    local_start: int
    local_end: int
    video_path: Path
    global_start: int
    global_end: int
    bbox_images: Path | None | IO[bytes] = None

    @property
    def length(self) -> int:
        return self.global_end - self.global_start


class Timer:
    """Simple class for measuring execution time during the whole process"""

    start_time: datetime | None = None
    finish_time: datetime | None = None

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.finish()

    def __init__(self, name: str = "") -> None:
        self.name = name

    def reset(self) -> None:
        self.start_time = None
        self.finish_time = None

    @property
    def interval(self) -> None | timedelta:
        if self.finish_time is None or self.start_time is None:
            return None
        return self.finish_time - self.start_time

    @property
    def started(self) -> bool:
        return self.start_time is not None

    @property
    def finished(self) -> bool:
        return self.interval is not None

    def start(self) -> None:
        logging.info(
            "[blue bold]START %s", self.name, extra={"markup": True}, stacklevel=2
        )
        self.start_time = datetime.now()

    def finish(self, raise_if_not_started=True) -> None:
        if not self.started and raise_if_not_started:
            raise RuntimeError("Timer finish method called before start method")

        self.finish_time = datetime.now()

        logging.info(
            f"[blue bold]FINISH {self.name}, it took {self}",
            extra={"markup": True},
            stacklevel=2,
        )

    def __str__(self) -> str:
        return str(self.interval or "Not finished").split(".")[0]

    @classmethod
    def from_dict(cls, d: dict) -> "Timer":
        obj = cls.__new__(cls)
        obj.name = d["name"]
        d.pop("py/object", None)

        if "interval" in d:  # v5.1.0 compatibility
            if d["start_time"] > 0:
                obj.start_time = datetime.fromtimestamp(d["start_time"])

            if d["interval"] > 0:
                obj.finish_time = datetime.fromtimestamp(
                    d["start_time"] + d["interval"]
                )

        else:
            if "start_time" in d:
                obj.start_time = datetime.fromisoformat(d["start_time"])
            if "finish_time" in d:
                obj.finish_time = datetime.fromisoformat(d["finish_time"])

        return obj


@dataclass(slots=True)
class LengthCalibration:
    "Length calibration used in the Validator to transform pixel units to user defined units."

    color: int = 0x000000
    point_A: Sequence[float] | None = None
    point_B: Sequence[float] | None = None
    distance: float | None = None

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls.__new__(cls)
        obj.point_A = d.get("point_A")
        obj.point_B = d.get("point_B")
        obj.distance = d.get("distance")
        return obj

    def value(self) -> float | None:
        if self.point_A is None or self.point_B is None or self.distance is None:
            return None
        return (
            sqrt(
                (self.point_A[0] - self.point_B[0]) ** 2
                + (self.point_A[1] - self.point_B[1]) ** 2
            )
            / self.distance
        )

    def add_point(self, point: Sequence[float]) -> None:
        if self.point_A is None:
            self.point_A = point
            return
        if self.point_B is None:
            self.point_B = point
            return
        raise ValueError("Calibration is already full")

    def completed(self) -> bool:
        return self.has_two_points() and self.distance is not None

    def has_two_points(self) -> bool:
        return self.point_A is not None and self.point_B is not None

    def __str__(self) -> str:
        return f"[{self.point_A}, {self.point_B}]: {self.distance}"


def assert_knowledge_transfer_is_possible(
    knowledge_transfer_folder: Path | None, n_animals: int
) -> list[int]:
    if knowledge_transfer_folder is None:
        raise IdtrackeraiError(
            "To perform knowledge/identity transfer you "
            "need to provide a path for the variable "
            "'KNOWLEDGE_TRANSFER_FOLDER'"
        )

    model_params_path = knowledge_transfer_folder / "model_params.json"
    if model_params_path.is_file():
        model_params_dict = json.load(model_params_path.open())
        n_classes, image_size = extract_parameters_from_model_json(model_params_dict)

    elif model_params_path.with_suffix(".npy").is_file():
        model_params_dict = np.load(
            model_params_path.with_suffix(".npy"), allow_pickle=True
        ).item()  # loading from v4
        n_classes, image_size = extract_parameters_from_model_json(model_params_dict)

    else:
        logging.warning('"%s" file not found', model_params_path)
        n_classes, image_size = extract_parameters_from_model_state_dict(
            knowledge_transfer_folder
        )

    if n_animals != n_classes:
        raise IdtrackeraiError(
            "Tracking with knowledge/identity transfer is not possible. "
            "The number of animals in the video needs to be the same as "
            "the number of animals in the transferred network."
        )

    logging.info(
        "Tracking with knowledge transfer. "
        "The identification image size will be matched "
        "to the image_size of the transferred network: %s",
        image_size,
    )
    return image_size


def extract_parameters_from_model_json(model_parameters: dict):
    image_size = model_parameters["image_size"]
    n_classes = (
        model_parameters["n_classes"]
        if "n_classes" in model_parameters  # 5.1.6 compatibility
        else model_parameters["number_of_classes"]
    )
    return n_classes, image_size


def extract_parameters_from_model_state_dict(knowledge_transfer_folder: Path):
    logging.info("Extracting model parameters from state dictionary")
    # this import is here (not at the top of the file) to avoid its loading process
    # when loading GUIs without identity_transfer (almost always)
    import torch

    model_dict_path = knowledge_transfer_folder / "identification_network.model.pth"
    model_state_dict: dict[str, torch.Tensor] = torch.load(model_dict_path)
    if "fc2.weight" in model_state_dict:
        layer_in_dimension = model_state_dict["fc1.weight"].size(1)
        n_classes = len(model_state_dict["fc2.weight"])
    else:
        layer_in_dimension = model_state_dict["layers.9.weight"].size(1)
        n_classes = len(model_state_dict["layers.11.weight"])
    image_size = int(4 * sqrt(layer_in_dimension / 100)) + 2
    return n_classes, [image_size, image_size, 1]


def pprint_dict(d: dict, name: str = "") -> str:
    text = f"[bold blue]{name}[/]:" if name else ""

    pad = min(max(map(len, d.keys())), 30)

    for key, value in d.items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value and isinstance(value[0], Path):
            value = list(map(str, value))
        if isinstance(value, Path):
            value = str(value)
        if len(repr(value)) < 50 or not isinstance(value, list):
            text += f"\n[bold]{key:>{pad}}[/]={repr(value)}"
        else:
            s = f"[{repr(value[0])}"
            for item in value[1:]:
                s += f",\n{' '*pad}  {repr(item)}"
            s += "]"
            text += f"\n[bold]{key:>{pad}}[/]={s}"
    return text


def load_id_images(
    id_images_file_paths: Sequence[Path | str],
    images_indices: Sequence[tuple[int, int]] | np.ndarray,
    verbose=True,
    dtype: Type[np.number] | None = None,
) -> np.ndarray:
    """Loads the identification images from disk.

    Parameters
    ----------
    id_images_file_paths : list
        List of strings with the paths to the files where the images are
        stored.
    images_indices : list
        List of tuples (image_index, episode) that indicate each of the images
        to be loaded

    Returns
    -------
    Numpy array
        Numpy array of shape [number of images, width, height]
    """
    img_indices, episodes = np.asarray(images_indices).T

    images = None

    for episode in track(
        np.unique(episodes), "Loading identification images from disk", verbose=verbose
    ):
        where = episodes == episode
        indices = img_indices[where]

        with h5py.File(id_images_file_paths[episode], "r") as file:
            if len(indices) > 100 or len(indices) > len(np.unique(indices)):
                # for more than 100 images, it's more efficient to load the entire file and select the desired indices
                # repetitions in indices is not acceptable for specific indices reading in h5py
                episode_imgs: np.ndarray = file["id_images"][:][indices]  # type: ignore
            else:
                # for less than 100 images, it's faster to get only the specific indices but h5py requires the indices to be sorted
                order = np.argsort(indices)
                unorder = np.empty_like(order)
                unorder[order] = np.arange(len(order))
                episode_imgs: np.ndarray = file["id_images"][indices[order]][unorder]  # type: ignore

        if images is None:
            # We take the first iteration to extract the image shape and dtype
            images = np.empty(
                (len(images_indices), *episode_imgs.shape[1:]),
                dtype or episode_imgs.dtype,
            )

        images[where] = episode_imgs

    if images is None:
        # in case of not having any iteration
        return np.zeros((0, 30, 30), np.uint8)

    return images


def json_default(obj):
    """Encodes non JSON serializable object as dicts"""
    match obj:
        case Path():
            return str(obj)
        case LengthCalibration():
            out = asdict(obj)
            out.pop("color")
            return out
        case Timer():
            return vars(obj)
        case np.integer():
            return int(obj)
        case np.floating():
            return float(obj)
        case np.ndarray():
            return {"py/object": "np.ndarray", "values": obj.tolist()}
        case set():
            return list(obj)
        case datetime():
            return obj.isoformat()
        case _:
            raise ValueError(f"Could not JSON serialize {obj} of type {type(obj)}")


def json_object_hook(d: dict):
    """Decodes dicts from `json_default`"""
    py_object = d.pop("py/object", None)
    if py_object is None:
        return d
    if py_object == "Path":
        return Path(d["path"])
    if py_object == "Episode":
        return Episode(**d)
    if py_object == "Timer":
        return Timer.from_dict(d)
    if py_object == "np.ndarray":
        return np.asarray(d["values"])
    if py_object == "set":
        return set(d["values"])
    raise ValueError(f"Could not read {d} of type {py_object}")


def resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def clean_attrs(obj: object):
    """Removes instances attributes if they are redundant
    with the class attributes"""
    class_attr = obj.__class__.__dict__

    attributes_to_remove: list[str] = [
        attr
        for attr, value in obj.__dict__.items()
        if attr in class_attr
        and isinstance(class_attr[attr], type(value))
        and class_attr[attr] == value
    ]

    for attr in attributes_to_remove:
        delattr(obj, attr)
