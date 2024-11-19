"A simple module to inspect contrastive clusters after tracking."
import argparse
import logging
from collections.abc import Sequence
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
    labels = []
    frames = []
    frag_indices = []
    for frag in frags.individual_fragments:
        if (
            len(frag) >= conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
            and frag.identity is not None
        ):
            locations += frag.image_locations
            labels += [frag.identity if frag.identity is not None else 0] * len(frag)
            frames += range(frag.start_frame, frag.end_frame)
            frag_indices += [frag.identifier] * len(frag)

    selection = np.arange(len(labels))
    logging.info(f"Collected {len(selection)} images in total")
    if images_per_id > 0 and len(selection) > images_per_id * frags.n_animals:
        images_per_id = int(images_per_id)
        logging.info(f"Using a subset of {images_per_id * frags.n_animals} images")
        np.random.shuffle(selection)
        selection = selection[: images_per_id * frags.n_animals]

    locations = np.asarray(locations)[selection]
    labels = np.asarray(labels)[selection]
    frames: Sequence[int] = np.asarray(frames)[selection].tolist()
    frag_indices = np.asarray(frag_indices)[selection]

    if gt_path is not None:
        # if there are groundtruth trajectories, lets get the labels from there
        groundtruth = load_trajectories(gt_path)["trajectories"]
        # we need the blob's centroids
        blobs = ListOfBlobs.load(session_path / "preprocessing/list_of_blobs.pickle")
        labels = []
        for frame, frag_idx in zip(frames, frag_indices):
            for blob in blobs.blobs_in_video[frame]:
                if blob.fragment_identifier == frag_idx:
                    labels.append(np.argmin(cdist([blob.centroid], groundtruth[frame])))
                    break
            else:
                raise RuntimeError(f"Blob with {frag_idx=} not found in {frame=}")
        labels = np.asarray(labels) + 1

    labels_for_plot = labels.astype(float)
    labels_for_plot[labels_for_plot == 0] = np.nan
    labels_for_plot -= 1
    labels_for_plot /= labels_for_plot.max() + 1

    data_loader = get_onthefly_dataloader(locations, frags.id_images_file_paths)

    embs = []
    for images, _labels in track(data_loader, "Embedding images"):
        embs.append(resnet(images.to(DEVICE)).numpy(force=True))
    embs = np.concatenate(embs)

    logging.info(f"Computing t-SNE with {len(embs)} data points")
    tsne = TSNE(n_jobs=4).fit_transform(embs)

    colormap = plt.get_cmap("hsv")
    colormap.set_bad("white")
    colors = colormap(labels_for_plot)

    fig = plt.figure(figsize=(8, 8))
    ax = Axes(fig, (0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.scatter(*tsne.T, c=colors, s=3, lw=0)
    ax.set(aspect=1)
    fig.savefig(save_folder / "t-SNE.png")

    np.savetxt(
        save_folder / "embeddings.csv",
        np.column_stack((embs, tsne, labels)),
        fmt=["%+.4f"] * (len(embs[0]) + 2) + ["%d"],
        delimiter=", ",
        header=", ".join(f"  dim_{i}" for i in range(8))
        + ",  t-SNE_1,  t-SNE_2"
        + ", identity",
        comments="",
    )
    logging.info(f"Plot and csv saved in {save_folder}")


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
            "identities are used to assign colors in the scatter plot and populate "
            "the identity column in the CSV output. The path can point to a "
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
