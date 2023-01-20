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
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from time import perf_counter

import cv2
import h5py
import numpy as np
from rich.progress import track


def round(arr):
    return np.rint(arr).astype(int)


### MKL
def set_mkl_to_single_thread():
    logging.info("Setting MKL library to use single thread")
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"


def set_mkl_to_multi_thread():
    logging.info("Setting MKL library to use multiple threads")
    os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["MKL_DYNAMIC"] = "TRUE"


### Object utils ###
def delete_attributes_from_object(object_to_modify, list_of_attributes):
    [
        delattr(object_to_modify, attribute)
        for attribute in list_of_attributes
        if hasattr(object_to_modify, attribute)
    ]


def interpolate_nans(t):
    """Interpolates nans linearly in a trajectory

    :param t: trajectory
    :returns: interpolated trajectory
    """
    shape_t = t.shape
    reshaped_t = t.reshape((shape_t[0], -1))
    for timeseries in range(reshaped_t.shape[-1]):
        y = reshaped_t[:, timeseries]
        nans, x = _nan_helper(y)  # type: ignore
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    # Ugly slow hack, as reshape seems not to return a view always
    back_t = reshaped_t.reshape(shape_t)
    t[...] = back_t


def _nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    assert False  # there's a warning on nonzero()
    return np.isnan(y), lambda z: z.nonzero()[0]


def create_dir(path: Path, remove_existing=False):
    if path.is_dir():
        if remove_existing:
            rmtree(path)
            path.mkdir()
            logging.info(f"Directory {path} has been cleaned")
        else:
            logging.info(f"Directory {path} already exists")
    else:
        path.mkdir()
        logging.info(f"Directory {path} has been created")


def remove_dir(path: Path):
    if path.is_dir():
        rmtree(path, ignore_errors=True)
        logging.info(f"Directory {path} has been removed")
    else:
        logging.info(f"Directory {path} not found, can't remove")


def remove_file(path: Path):
    if path.is_file():
        path.unlink()
        logging.info(f"File {path} has been removed")
    else:
        logging.info(f"File {path} not found, can't remove")


def assert_all_files_exist(paths: list[Path]):
    """Returns FileNotFoundError if any of the paths is not an existing file"""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"File {path} not found")


def get_vertices_from_label(label: str, close=False):
    """Transforms a string representation of a polygon from the
    ROI widget (idtrackerai_app) into a vertices np.array"""
    data = json.loads(label[10:].replace("'", '"'))

    if label[2:9] == "Polygon":
        vertices = np.asarray(data)
    elif label[2:9] == "Ellipse":
        x0, y0 = data["center"]
        a, b = data["axes"]
        angle = data["angle"]
        t = np.linspace(0, 2 * np.pi, 100)
        x = a * np.cos(t)
        y = b * np.sin(t)
        rot_x = np.cos(angle) * x - np.sin(angle) * y + x0
        rot_y = np.sin(angle) * x + np.cos(angle) * y + y0
        vertices = np.asarray([rot_x, rot_y]).T
    else:
        raise TypeError(label)

    if close:
        return np.vstack([vertices, vertices[0]]).astype(np.int32)
    else:
        return vertices.astype(np.int32)


def build_ROI_mask_from_list(width, height, list_of_ROIs):
    """Transforms a list of polygons (as type str) from
    ROI widget (idtrackerai_app) into a boolean np.array mask"""
    if not list_of_ROIs:
        return np.ones((height, width), bool)
    else:
        if isinstance(list_of_ROIs, str):
            list_of_ROIs = list(list_of_ROIs)
        ROI_mask = np.zeros((height, width), np.uint8)
        for line in list_of_ROIs:
            vertices = get_vertices_from_label(line)
            if line[0] == "+":
                cv2.fillPoly(ROI_mask, [vertices][::-1], color=1)
            elif line[0] == "-":
                cv2.fillPoly(ROI_mask, [vertices][::-1], color=0)
            else:
                raise TypeError
        return ROI_mask.astype(bool)


class CustomError(Exception):
    pass


@dataclass
class Episode:
    index: int
    local_start: int
    local_end: int
    video_path_index: int
    global_start: int
    global_end: int


class Timer:
    """Simple class for measuring execution time during the whole process"""

    has_finished: bool
    has_started: bool
    name: str

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.has_finished = False
        self.has_started = False
        self.interval = -1
        self.start_time = -1

    def start(self):
        logging.info("[blue bold]START " + self.name, extra={"markup": True})
        self.start_time = perf_counter()
        self.started = True

    def finish(self) -> float:
        if self.start_time == -1:
            raise RuntimeError("Timer finish method called before start method")

        self.interval = perf_counter() - self.start_time
        self.has_finished = True
        logging.info(
            f"[blue bold]FINISH {self.name}, it took {self}", extra={"markup": True}
        )
        return self.interval

    def __str__(self) -> str:
        if self.interval > 6000:
            return f"{self.interval/3600:.4f} hours"
        if self.interval > 100:
            return f"{self.interval/60:.4f} minutes"
        else:
            return f"{self.interval:.4f} seconds"

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls.__new__(cls)
        obj.__dict__.update(d)
        return obj


def check_if_identity_transfer_is_possible(
    number_of_animals: int, knowledge_transfer_folder: Path | None
) -> tuple[bool, list[int]]:
    if knowledge_transfer_folder is None:
        raise CustomError(
            "To perform identity transfer you "
            "need to provide a path for the variable "
            "'KNOWLEDGE_TRANSFER_FOLDER'"
        )

    kt_info_dict_path = knowledge_transfer_folder / "model_params.json"
    if kt_info_dict_path.is_file():
        knowledge_transfer_info_dict = json.loads(kt_info_dict_path.read_text())
        assert "image_size" in knowledge_transfer_info_dict
    else:
        raise CustomError(
            "To perform identity transfer the models_params.npy file "
            "is needed to check the input_image_size and "
            "the number_of_classes of the model to be loaded"
        )
    is_identity_transfer_possible = (
        number_of_animals == knowledge_transfer_info_dict["number_of_classes"]
    )
    if is_identity_transfer_possible:
        logging.info(
            "Tracking with identity transfer. "
            "The identification_image_size will be matched "
            "to the image_size of the transferred network"
        )
        id_image_size = knowledge_transfer_info_dict["image_size"]
    else:
        logging.warning(
            "Tracking with identity transfer is not possible. "
            "The number of animals in the video needs to be the same as "
            "the number of animals in the transferred network"
        )
        id_image_size = []

    return is_identity_transfer_possible, id_image_size


def pprint_dict(d: dict, name: str = "") -> str:
    text = f"[bold blue]{name}[/]:" if name else ""

    pad = min(max(map(len, d.keys())), 25)

    for key, value in d.items():
        if len(repr(value)) < 50 or not isinstance(value, (list, tuple)):
            text += f"\n[bold]{key:>{pad}}[/] = {repr(value)}"
        else:
            s = f"[{repr(value[0])}"
            for item in value[1:]:
                s += f",\n{' '*pad}    {repr(item)}"
            s += f"]"
            text += f"\n[bold]{key:>{pad}}[/] = {s}"
    return text


def load_id_images(
    id_images_file_paths: list[Path], images_indices: list[tuple[int, int]]
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
    hdf5_datasets: list[h5py.Dataset] = []
    for path in id_images_file_paths:
        dataset = h5py.File(path, "r")["id_images"]
        assert isinstance(dataset, h5py.Dataset)
        hdf5_datasets.append(dataset)

    # Create entire output array
    test_image = hdf5_datasets[images_indices[0][1]][images_indices[0][0]]
    images = np.empty((len(images_indices), *test_image.shape), test_image.dtype)

    # Fill the output array
    for i, (image, episode) in track(
        enumerate(images_indices),
        total=len(images_indices),
        description="Loading identification images from the disk",
    ):
        images[i] = hdf5_datasets[episode][image]

    for hdf5_dataset in hdf5_datasets:
        hdf5_dataset.file.close()

    return images


def json_default(obj):
    """Encodes non JSON serializable object as dicts"""
    if isinstance(obj, Path):
        return {"py/object": "Path", "path": str(obj)}

    if isinstance(obj, (Timer, Episode)):
        dic = {"py/object": obj.__class__.__name__}
        dic.update(obj.__dict__)
        return dic

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return {"py/object": "np.ndarray", "values": obj.tolist()}

    logging.error(f"Could not JSON serialize {obj} of type {type(obj)}")


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
    else:
        return d


def resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()
