# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
from shutil import rmtree
from typing import Iterable, Optional, TypeVar

import cv2
import h5py
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TimeRemainingColumn

from .init_logger import CustomError

InputType = TypeVar("InputType")


def track(
    sequence: Iterable[InputType],  # TODO also Sequence?
    desc: str = "Working...",
    total: Optional[float] = None,
) -> Iterable[InputType]:
    """A custom interpretation of rich.progress.track"""

    progress = Progress(
        " " * 18 + desc,
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
        stacklevel=3,
        extra={"markup": True},
    )


def delete_attributes_from_object(object_to_modify, list_of_attributes):
    for attribute in list_of_attributes:
        if hasattr(object_to_modify, attribute):
            delattr(object_to_modify, attribute)


def create_dir(path: Path, remove_existing=False):
    if path.is_dir():
        if remove_existing:
            rmtree(path)
            path.mkdir()
            logging.info(f"Directory {path} has been cleaned", stacklevel=3)
        else:
            logging.info(f"Directory {path} already exists", stacklevel=3)
    else:
        if not path.parent.is_dir():
            path.parent.mkdir()
        path.mkdir()
        logging.info(f"Directory {path} has been created", stacklevel=3)


def remove_dir(path: Path):
    if path.is_dir():
        rmtree(path, ignore_errors=True)
        logging.info(f"Directory {path} has been removed", stacklevel=3)
    else:
        logging.info(f"Directory {path} not found, can't remove", stacklevel=3)


def remove_file(path: Path):
    if path.is_file():
        path.unlink()
        logging.info(f"File {path} has been removed", stacklevel=3)


def assert_all_files_exist(paths: list[Path]):
    """Returns FileNotFoundError if any of the paths is not an existing file"""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"File {path} not found")


def get_vertices_from_label(label: str, close=False):
    """Transforms a string representation of a polygon from the
    ROI widget (idtrackerai_app) into a vertices np.array"""
    try:
        data = json.loads(label[10:].replace("'", '"'))
    except ValueError:
        raise CustomError(f'Not recognized ROI representation: "{label}"')

    if label[2:9] == "Polygon":
        vertices = np.asarray(data)
    elif label[2:9] == "Ellipse":
        vertices = cv2.ellipse2Poly(
            data["center"], data["axes"], data["angle"], 0, 360, 2
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
            cv2.fillPoly(ROI_mask, (vertices,), color=255)
        elif line[0] == "-":
            cv2.fillPoly(ROI_mask, (vertices,), color=0)
        else:
            raise TypeError
    return ROI_mask


@dataclass(slots=True)
class Episode:
    index: int
    local_start: int
    local_end: int
    video_path_index: int
    global_start: int
    global_end: int


class Timer:
    """Simple class for measuring execution time during the whole process"""

    start_time: datetime | None = None
    finish_time: datetime | None = None

    def __init__(self, name: str = ""):
        self.name = name

    def reset(self):
        self.start_time = None
        self.finish_time = None

    @property
    def interval(self):
        if self.finish_time is None or self.start_time is None:
            return None
        return self.finish_time - self.start_time

    @property
    def started(self):
        return self.start_time is not None

    @property
    def finished(self):
        return self.interval is not None

    def start(self):
        logging.info(
            "[blue bold]START %s", self.name, extra={"markup": True}, stacklevel=3
        )
        self.start_time = datetime.now()

    def finish(self, raise_if_not_started=True):
        if not self.started and raise_if_not_started:
            raise RuntimeError("Timer finish method called before start method")

        self.finish_time = datetime.now()

        logging.info(
            f"[blue bold]FINISH {self.name}, it took {self}",
            extra={"markup": True},
            stacklevel=3,
        )

    def __str__(self) -> str:
        return str(self.interval or "Not finished").split(".")[0]

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls.__new__(cls)
        obj.name = d["name"]

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


def assert_knowledge_transfer_is_possible(
    knowledge_transfer_folder: Path | None, n_animals: int
) -> list[int]:
    if knowledge_transfer_folder is None:
        raise CustomError(
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
        raise CustomError(
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

    pad = min(max(map(len, d.keys())), 25)

    for key, value in d.items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value and isinstance(value[0], Path):
            value = list(map(str, value))
        if isinstance(value, Path):
            value = str(value)
        if len(repr(value)) < 50 or not isinstance(value, list):
            text += f"\n[bold]{key:>{pad}}[/] = {repr(value)}"
        else:
            s = f"[{repr(value[0])}"
            for item in value[1:]:
                s += f",\n{' '*pad}    {repr(item)}"
            s += "]"
            text += f"\n[bold]{key:>{pad}}[/] = {s}"
    return text


def load_id_images(
    id_images_file_paths: list[Path], images_indices: Iterable[tuple[int, int]]
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
    if isinstance(images_indices, zip):
        images_indices = list(images_indices)

    img_indices, episodes = np.asarray(images_indices).T

    # Create entire output array
    with h5py.File(id_images_file_paths[0], "r") as file:
        test_dataset = file["id_images"]
        images = np.empty(
            (len(images_indices), *test_dataset.shape[1:]), test_dataset.dtype  # type: ignore
        )

    for episode in track(set(episodes), "Loading identification images from disk"):
        where = episodes == episode
        with h5py.File(id_images_file_paths[episode], "r") as file:
            images[where] = file["id_images"][:][img_indices[where]]  # type: ignore

    return images


def json_default(obj):
    """Encodes non JSON serializable object as dicts"""
    if isinstance(obj, Path):
        return {"py/object": "Path", "path": str(obj)}

    if isinstance(obj, (Timer, Episode)):
        return {"py/object": obj.__class__.__name__} | obj.__dict__

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return {"py/object": "np.ndarray", "values": obj.tolist()}

    if isinstance(obj, set):
        return {"py/object": "set", "values": list(obj)}

    if isinstance(obj, datetime):
        return obj.isoformat()

    raise ValueError(f"Could not JSON serialize {obj} of type {type(obj)}")


def json_object_hook(d: dict):
    """Decodes dicts from `json_default`"""
    if "py/object" in d:
        cls = d.pop("py/object")
        if cls == "Path":
            return Path(d["path"])
        if cls == "Episode":
            return Episode(**d)
        if cls == "Timer":
            return Timer.from_dict(d)
        if cls == "np.ndarray":
            return np.asarray(d["values"])
        if cls == "set":
            return set(d["values"])
        raise ValueError(f"Could not read {d}")
    return d


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
