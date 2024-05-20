"Contrastive Learning module"

import logging
import os
import random
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np
import psutil
import torch
from h5py import File
from rich.console import Console
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from torch import Tensor
from torch.nn.functional import pairwise_distance, relu
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torchvision.models.resnet import BasicBlock, ResNet

from idtrackerai import Fragment, GlobalFragment, ListOfFragments
from idtrackerai.base.network import DEVICE, get_onthefly_dataloader
from idtrackerai.utils import IdtrackeraiError, conf, load_id_images, track


class PairsOfFragments(Dataset):
    """Dataset with all pairs of fragments (positive and negative) which returns
    only the locations of selected images (the images are loaded in collate_fun)"""

    pairs: list[tuple[Fragment, Fragment]]

    def __init__(self, pairs: list[tuple[Fragment, Fragment]]) -> None:
        super().__init__()
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, pair_index: int
    ) -> tuple[tuple[int, int], tuple[int, int], int]:
        frag_A, frag_B = self.pairs[pair_index]
        img_index_A = random.randint(0, frag_A.n_images - 1)
        img_index_B = random.randint(0, frag_B.n_images - 1)

        return (
            (frag_A.images[img_index_A], frag_A.episodes[img_index_A]),
            (frag_B.images[img_index_B], frag_B.episodes[img_index_B]),
            pair_index,
        )


class BatchSampler(Sampler[list[int]]):
    """Custom implementation of a torch.utils.data.BatchSampler where the
    indices come from a probability distribution from self.negative_probabilities
    and self.positive_probabilities and the __iter__ method yield batches while
    the probabilities can be updated on the fly"""

    def __init__(
        self,
        negative_weights: Tensor,
        positive_weights: Tensor,
        negative_scores: Tensor,
        positive_scores: Tensor,
        batch_size: int,
        n_batches: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.update_probabilities(
            negative_weights, positive_weights, negative_scores, positive_scores
        )

    def __iter__(self) -> Iterator[list[int]]:
        if self.n_batches is None:
            raise RuntimeError(f"BatchSampler has {self.n_batches = }")
        for _ in range(self.n_batches):
            yield (
                torch.multinomial(
                    self.negative_probabilities, self.batch_size, replacement=True
                ).tolist()
                + (
                    torch.multinomial(
                        self.positive_probabilities, self.batch_size, replacement=True
                    )
                    + len(self.negative_probabilities)
                ).tolist()
            )

    def update_probabilities(
        self,
        negative_weights: Tensor,
        positive_weights: Tensor,
        negative_scores: Tensor,
        positive_scores: Tensor,
    ) -> None:
        self.negative_probabilities = (
            negative_weights + negative_scores / negative_scores.sum()
        )
        self.positive_probabilities = (
            positive_weights + positive_scores / positive_scores.sum()
        )


class ContrastiveDataLoader(Protocol):
    """Protocol class for better typing our DataLoaders"""

    batch_sampler: BatchSampler
    dataset: PairsOfFragments

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]: ...


def catch_out_of_memory(function: Callable):
    @wraps(function)
    def f(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            raise IdtrackeraiError(
                'GPU got out of memory. Decrease the "CONTRASTIVE_BATCHSIZE" parameter, '
                f"current value is {conf.CONTRASTIVE_BATCHSIZE}"
            ) from exc

    return f


@dataclass(slots=True)
class ContrastiveLearning:
    model: ResNet
    "RasNet18 model"
    optimizer: torch.optim.Optimizer
    "Optimizer"
    loaded_images: list[np.ndarray] | None
    "Identification images loaded in RAM to speedup training if enabled, else None"
    val_loader: ContrastiveDataLoader
    "Validation DataLoader"
    train_loader: ContrastiveDataLoader
    "Contrastive training DataLoader"
    gfrag_loader: ContrastiveDataLoader | None
    """DataLoader for a single Global Fragments images used to initialize kmeans clusters.
    None if there are no Global Fragments in the video."""

    penalties: Tensor
    """Sequence of floats representing the penalties of every pair of Fragments used in contrastive.
    Penalties increase when a pair of images is sampled from a specific pair of Fragments and its loss is non zero."""
    negative_weights: Tensor
    "The weights of every negative pair of Fragments related to their size. Used for sampling."
    positive_weights: Tensor
    "The weights of every positive pair of Fragments related to their size. Used for sampling."

    n_negative_pairs: int
    "The number of negative pairs of Fragments we have"
    check_every: int
    "Frequency of validation in training"

    n_animals: int
    "Number of animals in the video"

    first_epoch_to_validate: int
    "Quantity of epochs to skep before start validating"
    maximum_n_epochs: int
    "Limit of epochs before stopping training"
    learning_rate: float
    "Optimizer learning rate"
    embedding_dimensions: int
    "Number of dimensions of the embedded space"

    required_cluster_quality: float
    "Minimum size ratio (cluster quality measure) to stop training"

    saving_folder: Path
    "Saving folder for checkpoints"
    patience: int
    """Number of epochs with no improvements before stopping training"""
    batch_size: int
    """Number of pairs of each kind of images (positive and negative) used in a single training batch"""

    @property
    def negative_penalties(self) -> Tensor:
        return self.penalties[: self.n_negative_pairs]

    @property
    def positive_penalties(self) -> Tensor:
        return self.penalties[self.n_negative_pairs :]

    @property
    def model_checkpoint_path(self) -> Path:
        return self.saving_folder / "contrastive_checkpoint.pt"

    def __init__(
        self,
        fragments: ListOfFragments,
        saving_folder: Path,
        check_every: int = 1000,
        first_gfrag: GlobalFragment | None = None,
        batch_size: int = conf.CONTRASTIVE_BATCHSIZE,
        preload_images_max_mbytes: float | None = conf.CONTRASTIVE_MAX_MBYTES,
        min_frag_length: int = conf.MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION,
        learning_rate: float = 0.001,
        embedding_dimensions: int = 8,
        first_batch_group_to_check: int = 3,
        required_cluster_quality: float = 11,
        maximum_n_epochs: int = 1000,
        patience: int = 20,
    ) -> None:
        self.saving_folder = saving_folder
        self.first_epoch_to_validate = first_batch_group_to_check
        self.learning_rate = learning_rate
        self.embedding_dimensions = embedding_dimensions
        self.check_every = check_every
        self.required_cluster_quality = required_cluster_quality
        self.n_animals = fragments.n_animals
        self.maximum_n_epochs = maximum_n_epochs
        self.patience = patience
        self.batch_size = batch_size

        fragments_selection = [
            frag
            for frag in fragments
            if frag.is_an_individual and frag.n_images >= min_frag_length
        ]
        logging.info(
            f"Out of {len(fragments.fragments)} fragments, {len(fragments_selection)} "
            f"are individuals and longer than {min_frag_length-1} frames, they are gonna be used for contrastive training"
        )

        pairs_of_fragments: list[tuple[Fragment, Fragment]] = []
        negative_weights = []
        for fragment in fragments_selection:
            for coex_frag in fragment.coexisting_individual_fragments:
                if (
                    coex_frag.identifier > fragment.identifier
                    and coex_frag.is_an_individual
                    and coex_frag.n_images >= min_frag_length
                ):
                    pairs_of_fragments.append((fragment, coex_frag))
                    negative_weights.append(fragment.n_images + coex_frag.n_images)
        self.negative_weights = torch.tensor(negative_weights, dtype=torch.float64)
        self.negative_weights /= self.negative_weights.sum()

        self.positive_weights = torch.tensor(
            [frag.n_images for frag in fragments_selection], dtype=torch.float64
        )
        self.positive_weights /= self.positive_weights.sum()

        # add equal fragments
        first_equal_frag = len(pairs_of_fragments)
        pairs_of_fragments += ((frag, frag) for frag in fragments_selection)
        logging.info(
            f"Generated {first_equal_frag} negative and {len(pairs_of_fragments)-first_equal_frag} positive pairs of Fragments"
        )

        self.n_negative_pairs = len(self.negative_weights)

        self.penalties = torch.full([len(pairs_of_fragments)], 10, dtype=torch.double)

        self.loaded_images = self.preload_images(
            fragments.id_images_file_paths, preload_images_max_mbytes
        )
        self.build_dataloaders(
            pairs_of_fragments,
            fragments_selection,
            fragments.id_images_file_paths,
            1000 * self.n_animals,
            first_gfrag,
        )

    @staticmethod
    def preload_images(
        paths: Iterable[Path], size_limit: float | None
    ) -> None | list[np.ndarray]:
        if size_limit is None:
            size_limit = psutil.virtual_memory().available / (2 * 1024**2)
            logging.info(
                f"Size limit for pre-loading images not set, using half of the available memory in the system: {size_limit:.1f} MB"
            )

        n_megabytes = sum(
            File(path)["id_images"].nbytes  # type:ignore
            for path in paths
        ) / (1024 * 1024)
        logging.info(
            f"All identification images weight {n_megabytes:.1f} MB. The stated limit for them to be pre-loaded is {size_limit:.1f} MB"
        )

        if n_megabytes > size_limit:
            logging.info(
                "Not pre-loading identification images, they will be loaded from disk on the fly"
            )
            return None

        return [  # type:ignore
            File(path)["id_images"][:]  # type:ignore
            for path in track(paths, "Pre-loading all identification images to RAM")
        ]

    @staticmethod
    def criterion(
        embedded_A: Tensor,
        embedded_B: Tensor,
        first_positive: Tensor | int,
        margin: float = 10,
    ) -> Tensor:
        """Pairwise distance loss criterion.
        Negative pairs are pushed away until they are at distance `margin`.
        Positive pairs are pulled together until they are at distance 1"""
        distance = pairwise_distance(embedded_A, embedded_B)

        losses = torch.concatenate(  # negative first, positive after
            (margin - distance[:first_positive], distance[first_positive:] - 1)
        )
        return relu(losses).square()

    def build_dataloaders(
        self,
        pairs_of_fragments: list[tuple[Fragment, Fragment]],
        fragments_selection: Iterable[Fragment],
        id_images_file_paths: Sequence[Path],
        max_n_val_images: int,
        first_gfrag: GlobalFragment | None = None,
    ) -> None:

        train_dataset = PairsOfFragments(pairs_of_fragments)

        val_images = []
        for frag in fragments_selection:
            val_images += frag.image_locations

        if len(val_images) > max_n_val_images:
            rng = np.random.default_rng()
            val_images = rng.choice(val_images, max_n_val_images, replace=False)

        logging.info(f"Validating contrastive clusters with {len(val_images)} images")
        val_dataset = TensorDataset(
            torch.tensor(val_images), torch.zeros(len(val_images))
        )

        collate_fn = partial(
            collate_fun,
            id_images_paths=id_images_file_paths,
            loaded_images=self.loaded_images,
        )

        val_collate_fn = partial(
            val_collate_fun,
            id_images_paths=id_images_file_paths,
            loaded_images=self.loaded_images,
        )

        if self.loaded_images is None:
            # if images are not loaded, we need more workers to load them on the fly
            num_workers = 6
        else:
            # Windows copies the memory of the main process to all parallel workers.
            # So if we are dealing with preloaded images we don't want many workers
            num_workers = 1 if os.name == "nt" else 3

        self.val_loader = DataLoader(  # type:ignore
            dataset=val_dataset,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=val_collate_fn,
        )

        self.train_loader = DataLoader(  # type:ignore
            dataset=train_dataset,
            num_workers=num_workers,
            batch_sampler=BatchSampler(
                self.negative_weights,
                self.positive_weights,
                self.negative_penalties,
                self.positive_penalties,
                self.batch_size,
            ),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if first_gfrag is None:
            logging.info(
                'Initializing kmeans clusters from "k-means++"'
                "because there are no Global Fragments"
            )
            self.gfrag_loader = None
            return

        if first_gfrag.min_n_images_per_fragment < 30:
            logging.info(
                'Initializing kmeans clusters from "k-means++" because the '
                f"biggest Global Fragment ({first_gfrag.min_n_images_per_fragment}"
                " frames) is not big enough (30 frames)"
            )
            self.gfrag_loader = None
            return

        image_locations = []
        frag_ids = []

        for frag_id, fragment in enumerate(first_gfrag):
            image_locations += fragment.image_locations
            frag_ids += [frag_id] * fragment.n_images
        first_gfrag_dataset = TensorDataset(
            torch.tensor(image_locations), torch.tensor(frag_ids)
        )

        self.gfrag_loader = DataLoader(  # type:ignore
            dataset=first_gfrag_dataset,
            num_workers=num_workers,
            batch_size=self.batch_size,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=val_collate_fn,
        )

    def reset_model(self) -> None:
        model = ResNet(  # ResNet18
            BasicBlock, [2, 2, 2, 2], num_classes=self.embedding_dimensions
        )
        # adapt first conv layer to our single channel images (not RGB)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model = model.to(DEVICE)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    @catch_out_of_memory
    def train(self, reset_model: bool = True) -> None:
        "Main method to train the contrastive"
        if reset_model:
            self.reset_model()

        self.model.train()
        best_quality: float | np.float_ = 0
        steps_without_improvement: int = 0
        with Console().status("Training contrastive") as status:
            for epoch in range(self.maximum_n_epochs):
                start = perf_counter()

                self.train_step(
                    n_batches=self.check_every,
                    output=status.update,
                    starting_batch_number=epoch * self.check_every + 1,
                )
                stop = perf_counter()

                if epoch < self.first_epoch_to_validate:
                    continue

                status.update("Validating")

                cluster_quality = self.validate()

                status.stop()
                logging.debug(
                    f"Batch {(epoch+1)*self.check_every}: "
                    f"{self.check_every/(stop-start):5.1f} batches/s, cluster quality {cluster_quality:5.2f}"
                    f" (stop training when >= {self.required_cluster_quality})"
                )
                status.start()

                if cluster_quality > best_quality:
                    best_quality = cluster_quality
                    torch.save(self.model.state_dict(), self.model_checkpoint_path)
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if steps_without_improvement > self.patience:
                    logging.warning(
                        f"The model has not improved for {self.patience} steps, we stop the training"
                    )
                    self.model.load_state_dict(torch.load(self.model_checkpoint_path))
                    break

                if cluster_quality > self.required_cluster_quality:
                    break
            else:
                logging.warning(
                    "Maximum number of epochs reached, loading the best checkpoint"
                )
                self.model.load_state_dict(torch.load(self.model_checkpoint_path))

    @torch.inference_mode()
    def validate(self) -> np.float_:
        """Clustering images from self.val_loader and return the cluster quality
        (the minimal distance between cluster centers divided by the 90% percentile
        of the distance of images to their cluster center."""
        self.model.eval()
        embeddings = np.concatenate(
            [
                self.model.forward(images.to(DEVICE)).numpy(force=True)
                for (images, _labels) in self.val_loader
            ]
        )
        kmeans = MiniBatchKMeans(self.n_animals, **self.kmeans_init())
        distances = kmeans.fit_transform(embeddings)
        assignments = distances.argmin(1, keepdims=True)

        inner_distances = np.take_along_axis(distances, assignments, axis=1)
        outer_distances = kmeans.transform(kmeans.cluster_centers_)

        np.fill_diagonal(outer_distances, np.inf)
        min_outer_distances = outer_distances.min()

        return min_outer_distances / np.percentile(inner_distances, 90)

    def train_step(
        self,
        n_batches: int,
        output: Callable[[str], None] = print,
        starting_batch_number: int = 0,
    ) -> None:

        # this will make the dataloader to iterate n_batches times
        self.train_loader.batch_sampler.n_batches = n_batches

        self.model.train()
        for batch_number, (images_A, images_B, pair_indices) in enumerate(
            self.train_loader, starting_batch_number
        ):
            # each batch has batch_size negative pairs with
            #     images_A[: self.batch_size] -> images_B[: self.batch_size]
            # and batch_size positive pairs with
            #     images_A[self.batch_size:] -> images_B[self.batch_size:]
            # So, in each batch ResNet sees 4*batch_size images

            images_A = images_A.to(DEVICE, non_blocking=True)
            images_B = images_B.to(DEVICE, non_blocking=True)
            embedded_A = self.model.forward(images_A)
            embedded_B = self.model.forward(images_B)
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.criterion(embedded_A, embedded_B, self.batch_size)
            losses.mean().backward()
            self.optimizer.step()

            has_loss = (losses.detach() != 0).cpu()
            n_loss_negative = losses[: self.batch_size].count_nonzero().item()
            n_loss_positive = losses[self.batch_size :].count_nonzero().item()

            self.penalties += pair_indices.bincount(
                has_loss, minlength=len(self.penalties)
            )

            self.penalties *= 0.98

            self.train_loader.batch_sampler.update_probabilities(
                self.negative_weights,
                self.positive_weights,
                self.negative_penalties,
                self.positive_penalties,
            )

            output(
                f"[red]Batch {batch_number:2}: sampled {self.batch_size} positive pairs ({n_loss_positive} with loss) and "
                f"{self.batch_size} negative pairs ({n_loss_negative} with loss)"
            )

    @torch.inference_mode()
    def predict(
        self, fragments: ListOfFragments, first_gfrag: GlobalFragment | None = None
    ) -> None:
        image_locations: list[tuple[int, int]] = []
        lengths: list[int] = []

        individual_fragments = list(fragments.individual_fragments)

        for fragment in individual_fragments:
            image_locations += fragment.image_locations
            lengths.append(fragment.n_images)

        assert image_locations

        logging.debug(
            "Predicting %d images with contrastive model", len(image_locations)
        )

        dataloader = get_onthefly_dataloader(
            image_locations, fragments.id_images_file_paths
        )

        self.model.eval()
        embeddings = np.concatenate(
            [
                self.model.forward(images.to(DEVICE) / 255).numpy(force=True)
                for images, _labels in track(dataloader, "Predicting")
            ]
        )

        kmeans = MiniBatchKMeans(self.n_animals, **self.kmeans_init()).fit(embeddings)
        distances = kmeans.transform(embeddings)

        prob: np.ndarray = np.reciprocal(distances + 0.01) ** 7
        prob /= prob.sum(1, keepdims=True)

        assignments: np.ndarray = prob.argmax(1, keepdims=True)
        probabilities = np.take_along_axis(prob, assignments, axis=1)

        logging.debug("Computing fragment prediction statistics")

        fragments_assignments = np.split(
            assignments.flatten() + 1, np.cumsum(lengths)[:-1]
        )
        fragments_probabilities = np.split(
            probabilities.flatten(), np.cumsum(lengths)[:-1]
        )

        if first_gfrag is not None:
            # if there is a first Global Fragment, it should be already assigned with "temporary_id" from identity transfer or exclusive ROIs...
            # We will adapt cluster assignments to these identities
            translation_matrix = np.empty((self.n_animals, self.n_animals), int)
            for frag in first_gfrag:
                assert frag.temporary_id is not None
                translation_matrix[frag.temporary_id] = np.bincount(
                    fragments_assignments[individual_fragments.index(frag)] - 1,
                    minlength=self.n_animals,
                )

            ids_map = linear_sum_assignment(translation_matrix.T, maximize=True)[1]

            if not np.array_equal(ids_map, np.arange(self.n_animals)):
                logging.info(
                    "Applying previously assigned identities from the first Global Fragment to contrastive clusters"
                )
                assignments = np.vectorize(lambda x: ids_map[x])(assignments)
                fragments_assignments = np.split(
                    assignments.flatten() + 1, np.cumsum(lengths)[:-1]
                )

        for predictions, probabilities, fragment in zip(
            fragments_assignments,
            fragments_probabilities,
            fragments.individual_fragments,
        ):
            fragment.compute_identification_statistics(
                predictions, probabilities, self.n_animals
            )

    @torch.inference_mode()
    def kmeans_init(self) -> dict[str, Any]:
        # batch size should be proportional to the number of clusters but also not too small.
        # Also, in Windows, scikit-learn recommends a batch size greater than a specific expression, from `MiniBatchKMeans._warn_mkl_vcomp`
        batch_size = max(
            1024, 32 * self.n_animals, _openmp_effective_n_threads() * CHUNK_SIZE
        )

        if self.gfrag_loader is None:
            return {"batch_size": batch_size, "n_init": 20, "init": "k-means++"}

        embeddings = []
        labels = []

        self.model.eval()
        for images, labels_ in self.gfrag_loader:
            embeddings.append(self.model.forward(images.to(DEVICE)).numpy(force=True))
            labels.append(labels_.numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        cluster_centers = []
        for label in range(self.n_animals):
            cluster_centers.append(embeddings[labels == label].mean(0))
        return {
            "batch_size": batch_size,
            "n_init": 1,
            "init": np.asarray(cluster_centers),
        }


def val_collate_fun(
    batch: list[tuple[Tensor]],
    id_images_paths: Sequence[Path],
    loaded_images: list[np.ndarray] | None = None,
) -> list[Tensor]:
    """Receives the batch images locations (episode and index).
    These are used to load the images and generate the batch tensor"""
    locations, label = zip(*batch)
    locations = torch.stack(locations).numpy()

    return [load_images(locations, id_images_paths, loaded_images), torch.tensor(label)]


def collate_fun(
    batch: list[tuple[tuple[int, int], tuple[int, int], int]],
    id_images_paths: Sequence[Path],
    loaded_images: list[np.ndarray] | None = None,
) -> list[Tensor]:
    """Receives the batch images locations (episode and index).
    These are used to load the images and generate the batch tensor"""
    locations_A, locations_B, pair_indices = zip(*batch)
    images = load_images(locations_A + locations_B, id_images_paths, loaded_images)
    return [
        images[: len(locations_A)],
        images[len(locations_A) :],
        torch.tensor(pair_indices),
    ]


def load_images(
    image_locations: Sequence[tuple[int, int]] | np.ndarray,
    id_images_paths: Sequence[Path],
    loaded_images: list[np.ndarray] | None = None,
) -> Tensor:
    if loaded_images is None:
        # there are no preloaded images, lets get them from disk
        images = load_id_images(
            id_images_paths, image_locations, verbose=False, dtype=np.float32
        )
    else:
        # images are in RAM
        img_indices, episodes = np.asarray(image_locations).T
        images = np.empty((len(img_indices), *loaded_images[0].shape[1:]), np.float32)

        for episode in np.unique(episodes):
            where = episodes == episode
            images[where] = loaded_images[episode][img_indices[where]]

    return torch.from_numpy(images).contiguous().unsqueeze(1) / 255
