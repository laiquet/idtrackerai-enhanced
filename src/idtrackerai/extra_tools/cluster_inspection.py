"A simple module to inspect contrastive clusters after tracking, see :ref:`cluster inspection`."

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

from idtrackerai import IdtrackeraiError, ListOfBlobs, ListOfFragments, conf
from idtrackerai.base.network import DEVICE, ResNet18, get_onthefly_dataloader
from idtrackerai.utils import (
    load_trajectories,
    manage_exception,
    resolve_path,
    track,
    wrap_entrypoint,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


@torch.inference_mode()
def inspect_clusters(
    session_path: Path, images_per_id: float = 500, gt_path: Path | None = None
) -> None:
    """Use the trained ResNet to embed all images used in contrastive
    training or a subset of them, compute the t-SNE representation in
    2D and save the results in a .png scatter plot and a .csv file.

    Parameters
    ----------
    session_path : Path
        _description_
    images_per_id : int, optional
        _description_, by default 500

    Raises
    ------
    IdtrackeraiError
        _description_
    """
    plt.style.use("dark_background")

    session_path = resolve_path(session_path)
    logging.info(f"Computing t-SNE of {session_path}")

    frags = ListOfFragments.load(
        session_path / "preprocessing/list_of_fragments.json", reconnect=False
    )

    for file in ("identifier_contrastive.model.pt", "contrastive_checkpoint.pt"):
        try:
            resnet = ResNet18.from_file(session_path / "accumulation" / file).to(DEVICE)
        except FileNotFoundError:
            continue
        else:
            resnet.eval()
            break
    else:
        raise IdtrackeraiError(
            f"Could not find the contrastive model weights in the session folder {session_path}"
        )

    save_folder = session_path / "cluster_inspection"
    save_folder.mkdir(exist_ok=True)

    locations = []
    predicted_labels = []
    frames = []
    frag_indices = []
    for frag in frags.individual_fragments:
        if len(frag) >= conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION:
            locations += frag.image_locations
            predicted_labels += [
                frag.identity if frag.identity is not None else 0
            ] * len(frag)
            frames += range(frag.start_frame, frag.end_frame)
            frag_indices += [frag.identifier] * len(frag)

    selection = np.arange(len(predicted_labels))
    logging.info(f"Collected {len(selection)} images in total")
    if images_per_id > 0 and len(selection) > images_per_id * frags.n_animals:
        images_per_id = int(images_per_id)
        logging.info(f"Using a subset of {images_per_id * frags.n_animals} images")
        np.random.shuffle(selection)
        selection = selection[: images_per_id * frags.n_animals]

    locations = np.asarray(locations)[selection]
    predicted_labels = np.asarray(predicted_labels)[selection]
    frames = np.asarray(frames)[selection]
    frag_indices = np.asarray(frag_indices)[selection]

    # sort images by frame to speed up loading
    order = np.argsort(frames)
    locations = locations[order]
    predicted_labels = predicted_labels[order]
    frames = frames[order]
    frag_indices = frag_indices[order]

    data_loader = get_onthefly_dataloader(locations, frags.id_images_file_paths)

    embs = []
    for images, _labels in track(data_loader, "Embedding images"):
        embs.append(resnet(images.to(DEVICE)).numpy(force=True))
    embs = np.concatenate(embs)

    logging.info(f"Computing t-SNE with {len(embs)} data points")
    tsne = TSNE(n_jobs=4).fit_transform(embs)

    scatter_plot(
        *tsne.T, labels=predicted_labels, file=save_folder / "predicted_t-SNE.png"
    )

    if gt_path is not None:
        # if there are groundtruth trajectories, lets get the labels from there
        groundtruth = load_trajectories(gt_path)["trajectories"]
        # we need the blob's centroids
        blobs = ListOfBlobs.load(session_path / "preprocessing/list_of_blobs.pickle")
        gt_labels = []
        for frame, frag_idx in zip(frames, frag_indices):
            assert isinstance(frame, (np.integer, int))  # for Pylance
            for blob in blobs.blobs_in_video[frame]:
                if blob.fragment_identifier == frag_idx:
                    gt_labels.append(
                        np.argmin(cdist([blob.centroid], groundtruth[frame]))
                    )
                    break
            else:
                raise RuntimeError(f"Blob with {frag_idx=} not found in {frame=}")
        gt_labels = np.asarray(gt_labels) + 1

        scatter_plot(
            *tsne.T, labels=gt_labels, file=save_folder / "groundtruth_t-SNE.png"
        )

    else:
        gt_labels = predicted_labels * 0

    np.savetxt(
        save_folder / "embeddings.csv",
        np.column_stack((frag_indices, embs, tsne, predicted_labels, gt_labels)),
        fmt=["%4d"] + ["%+.4f"] * (len(embs[0]) + 2) + ["%d"] * 2,
        delimiter=", ",
        header="fragment, "
        + ", ".join(f"  dim_{i}" for i in range(8))
        + ",  t-SNE_1,  t-SNE_2"
        + ", predicted_id, groundtruth_id",
        comments="",
    )
    logging.info(f"Plot and csv saved in {save_folder}")


def scatter_plot(x: np.ndarray, y: np.ndarray, labels: np.ndarray, file: Path | str):
    """Expected Numpy array with IDs from 1 to n_animals, 0 values meaning unknown"""
    colormap = plt.get_cmap("hsv")
    colormap.set_bad("white")
    colormap.set_under("white")

    labels = labels.astype(float)
    labels -= 1
    norm = labels.max() + 1
    if norm > 0:
        # hsv colormap is circular, min and max values have the same color
        labels /= labels.max() + 1

    fig = plt.figure(figsize=(8, 8))
    ax = Axes(fig, (0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.scatter(x, y, c=colormap(labels), s=3, lw=0)
    ax.set(aspect=1)
    fig.savefig(file)


@wrap_entrypoint
def idtrackerai_inspect_clusters_entrypoint():
    argparser = argparse.ArgumentParser(
        description=(
            "Use the trained contrastive network (ResNet) to compute "
            "and store the image embeddings. It then generates a "
            "scatter plot of their t-SNE representation, enabling "
            "visual inspection of the resulting clusters. The results "
            "are saved in 'session_folder/cluster_inspection'"
        )
    )
    argparser.add_argument(
        "session_paths",
        help=(
            "Session paths to check the t-SNE on. "
            "Multiple session paths can be provided."
        ),
        type=Path,
        nargs="+",
    )
    argparser.add_argument(
        "--images_per_id",
        help=(
            "Sets the maximum number of images per animal to sample for speeding "
            "up t-SNE computation. The default value is 500. To disable subsampling, "
            "set this option to 'inf'. Note that subsampling is performed randomly "
            "across all identities, so the exact number of images per class may vary."
        ),
        type=float,
        default=500,
    )
    argparser.add_argument(
        "--gt_path",
        help=(
            "Specifies the path to the ground truth trajectories used to compare "
            "image centroids and extract their ground truth identities. These "
            "identities are used to assign colors in the groundtruth scatter plot and populate "
            "the groundtruth_id column in the CSV output. The path can point to a "
            "trajectory file or a session folder."
        ),
        type=Path,
    )
    args = argparser.parse_args()
    for path in args.session_paths:
        try:
            inspect_clusters(
                path, images_per_id=args.images_per_id, gt_path=args.gt_path
            )
        except Exception as exc:
            manage_exception(exc)


if __name__ == "__main__":
    idtrackerai_inspect_clusters_entrypoint()
