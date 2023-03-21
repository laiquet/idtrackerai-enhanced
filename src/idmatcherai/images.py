from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

NUM_IMAGES_PER_INDIVIDUAL = 3000


def extract_images_and_labels_from_list(identification_images_file_paths):
    info = {}
    all_images = []
    all_labels = []
    for id, identification_images_file_path in enumerate(
        identification_images_file_paths
    ):
        print("Extracting images from {}".format(identification_images_file_path))
        images, labels, inf = extract_images_and_labels(identification_images_file_path)
        info[str(id)] = inf
        labels = [id] * len(images)
        all_images.extend(images)
        all_labels.extend(labels)
        print("{} images and {} labels".format(len(images), len(labels)))

    return all_images, all_labels, info


def extract_images_and_labels(
    id_images_file_path: Path,
) -> tuple[None | np.ndarray, None | np.ndarray]:
    with h5py.File(id_images_file_path, "r") as file:
        id_images = (
            file["id_images"][:]
            if "id_images" in file
            else file["identification_images"][:]
        )
        assert isinstance(id_images, np.ndarray)

        if id_images.shape[0] == 0:
            # empty episode, v4 produces these episodes
            raise ValueError
        identities = file["identities"][:]
        assert isinstance(identities, np.ndarray)

        good_images = np.where(~np.isnan(identities))[0]
        if len(good_images) == 0:
            # not any identified animal
            raise ValueError

        return id_images[good_images], identities[good_images]

        # elif id_images.shape[0] != 0:
        #     p = np.clip(NUM_IMAGES_PER_INDIVIDUAL / id_images.shape[0], 0.0, 1.0)
        #     return [im for im in id_images if np.random.rand() < p], []

        # else:
        #     return [], []


def extact_all_images_and_labels(id_images_file_paths: Iterable[Path]):
    images = []
    labels = []
    for path in id_images_file_paths:
        try:
            episode_images, episode_labels = extract_images_and_labels(path)
            images.append(episode_images)
            labels.append(episode_labels)
        except (ValueError, AssertionError):
            pass

    return np.concatenate(images, axis=0), np.squeeze(np.concatenate(labels, axis=0))
