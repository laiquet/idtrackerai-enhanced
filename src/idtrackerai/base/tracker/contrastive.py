"Contrastive Learning module"

import logging
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator, Protocol, Sequence

import numpy as np
import torch
from h5py import File
from rich.console import Console
from rich.status import Status
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torchvision.models.resnet import BasicBlock, ResNet

from idtrackerai import Fragment, ListOfFragments
from idtrackerai.base.network import DEVICE, get_onthefly_dataloader
from idtrackerai.utils import load_id_images, track


class PairsOfFragments(Dataset):
    """Dataset with all pairs of fragments (positive and negative) which returns
    only the indices of selected images (proper images are loaded in collate_fun)"""

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
    indices come from a probability distribution from self.weights and the
    __iter__ method yield batches while the self.weights can be updated on the fly"""

    def __init__(
        self, weights: Tensor, batch_size: int, n_batches: int | None = None
    ) -> None:
        self.weights = weights
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self) -> Iterator[list[int]]:
        if self.n_batches is None:
            raise RuntimeError(f"BatchSampler has {self.n_batches = }")
        for _ in range(self.n_batches):
            yield torch.multinomial(
                self.weights, self.batch_size, replacement=True
            ).tolist()


class ContrastiveDataLoader(Protocol):
    """Protocol class for better typing our DataLoaders"""

    batch_sampler: BatchSampler
    dataset: PairsOfFragments

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]: ...


@dataclass(slots=True)
class ContrastiveLearning:
    model: ResNet
    optimizer: torch.optim.Optimizer
    loaded_images: list[np.ndarray] | None
    val_loader: ContrastiveDataLoader
    train_loader: ContrastiveDataLoader
    penalties: Tensor
    positive_err_rate: float
    negative_err_rate: float

    negative_weights: Tensor
    positive_weights: Tensor

    n_negative_pairs: int
    check_every: int

    n_animals: int

    first_epoch_to_validate: int
    maximum_n_epochs: int
    learning_rate: float
    embedding_dimensions: int

    cluter_centers: np.ndarray
    required_size_ratio: float

    saving_folder: Path

    @property
    def negative_penalties(self) -> Tensor:
        return self.penalties[: self.n_negative_pairs]

    @property
    def positive_penalties(self) -> Tensor:
        return self.penalties[self.n_negative_pairs :]

    @property
    def model_checkpoint_path(self) -> Path:
        return self.saving_folder / "model_checkpoint.pt"

    def __init__(
        self,
        fragments: ListOfFragments,
        saving_folder: Path,
        check_every: int = 1000,
        batch_size: int = 500,
        preload_images_max_mbytes: float = 0,
        min_frag_length: int = 4,
        learning_rate: float = 0.001,
        embedding_dimensions: int = 8,
        first_batch_group_to_check: int = 3,
        required_size_ratio: float = 11,
        maximum_n_epochs: int = 1000,
    ) -> None:
        self.saving_folder = saving_folder
        self.first_epoch_to_validate = first_batch_group_to_check
        self.learning_rate = learning_rate
        self.embedding_dimensions = embedding_dimensions
        self.check_every = check_every
        self.required_size_ratio = required_size_ratio
        self.n_animals = fragments.n_animals
        self.maximum_n_epochs = maximum_n_epochs

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

        self.positive_err_rate = 1.0
        self.negative_err_rate = 1.0
        self.penalties = torch.full([len(pairs_of_fragments)], 10, dtype=torch.double)

        self.preload_images(fragments.id_images_file_paths, preload_images_max_mbytes)
        self.build_dataloaders(
            pairs_of_fragments,
            fragments_selection,
            batch_size,
            fragments.id_images_file_paths,
            1000 * self.n_animals,
        )

    def preload_images(self, paths: Iterable[Path], size_limit: float) -> None:
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
            self.loaded_images = None
        else:
            self.loaded_images = [  # type:ignore
                File(path)["id_images"][:]  # type:ignore
                for path in track(paths, "Pre-loading all identification images to RAM")
            ]

    @staticmethod
    def criterion(
        embedded_A: Tensor, embedded_B: Tensor, positive: Tensor, margin: float = 10
    ) -> Tensor:
        """Pairwise distance loss criterion."""
        distance = (embedded_A - embedded_B).square().sum(1).sqrt()

        losses = torch.empty_like(distance)
        losses[positive] = distance[positive] - 1
        losses[~positive] = margin - distance[~positive]
        return torch.nn.functional.relu(losses).square()

    def build_dataloaders(
        self,
        pairs_of_fragments: list[tuple[Fragment, Fragment]],
        fragments_selection: Iterable[Fragment],
        batch_size: int,
        id_images_file_paths: Sequence[Path],
        max_n_val_images: int,
    ) -> None:

        train_dataset = PairsOfFragments(pairs_of_fragments)

        val_images = []
        for frag in fragments_selection:
            val_images += frag.image_locations

        if len(val_images) > max_n_val_images:
            rng = np.random.default_rng()
            val_images = rng.choice(val_images, max_n_val_images, replace=False)

        logging.info(f"Validating contrastive clusters with {len(val_images)} images")
        val_dataset = TensorDataset(torch.tensor(val_images))

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

        # if images are not loaded, we need more workers to load them on the fly
        num_workers = 6 if self.loaded_images is None else 3

        self.val_loader = DataLoader(  # type:ignore
            dataset=val_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=val_collate_fn,
        )

        self.train_loader = DataLoader(  # type:ignore
            dataset=train_dataset,
            num_workers=num_workers,
            batch_sampler=BatchSampler(
                weights=get_weights(
                    self.negative_weights,
                    self.positive_weights,
                    self.negative_penalties,
                    self.positive_penalties,
                ),
                batch_size=batch_size,
            ),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def reset_model(self) -> None:
        model = ResNet(  # ResNet18
            BasicBlock, [2, 2, 2, 2], num_classes=self.embedding_dimensions
        )
        model.conv1 = torch.nn.Conv2d(  # adapt first conv layer to our single channel images (not RGB)
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model = model.to(DEVICE)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def train(self, reset_model: bool = True) -> None:
        "Main method to train the contrastive"
        if reset_model:
            self.reset_model()

        self.model.train()
        start = perf_counter()
        best_ratio: float | np.float_ = 0
        with Console().status("Training contrastive") as status:
            for epoch in range(self.maximum_n_epochs):
                start = perf_counter()

                self.train_step(
                    status,
                    n_batches=self.check_every,
                    starting_batch=epoch * self.check_every,
                )
                stop = perf_counter()

                if epoch < self.first_epoch_to_validate:
                    continue

                status.update("Validating")

                size_ratio, other_data = self.validate()

                status.stop()
                logging.debug(
                    f"Batch: {epoch*self.check_every}-{(epoch+1)*self.check_every} "
                    f"{self.check_every/(stop-start):5.1f} batches/s | {size_ratio = :.2f}"
                    f" (stop training when >= {self.required_size_ratio}) | {other_data}"
                )
                status.start()

                if size_ratio > best_ratio:
                    best_ratio = size_ratio
                    torch.save(self.model.state_dict(), self.model_checkpoint_path)

                if size_ratio > self.required_size_ratio:
                    break
            else:
                logging.warning(
                    "Maximum number of epochs reached, loading the best checkpoint"
                )
                self.model.load_state_dict(torch.load(self.model_checkpoint_path))

    @torch.inference_mode()
    def validate(self) -> tuple[np.float_, Any]:
        "Clustering images from self.val_loader and return the 90% percentile of the distance to the closest cluster."
        self.model.eval()
        embeddings = np.concatenate([
            self.model.forward(images.to(DEVICE)).numpy(force=True)
            for (images,) in self.val_loader
        ])

        kmeans = MiniBatchKMeans(self.n_animals, n_init=20).fit(embeddings)
        distances = kmeans.transform(embeddings)

        prob: np.ndarray = np.reciprocal(distances + 0.01) ** 7
        prob /= prob.sum(1, keepdims=True)

        assignments = prob.argmax(1, keepdims=True)
        probabilities = np.take_along_axis(prob, assignments, axis=1)

        cluster_distances = np.take_along_axis(distances, assignments, axis=1)
        cluster_sizes = np.bincount(assignments.flatten())
        cluster_sizes.sort()

        outer_distances = pdist(kmeans.cluster_centers_).min(0)

        other_data = (
            cluster_sizes[0],
            cluster_sizes[-1],
            probabilities.mean(),
            outer_distances.mean(),
            np.percentile(cluster_distances, 90),
        )

        return outer_distances.mean() / np.percentile(cluster_distances, 90), other_data

    def train_step(
        self, status: Status, n_batches: int, starting_batch: int = 0
    ) -> None:

        # this will amke the dataloader to iterate n_batches times
        self.train_loader.batch_sampler.n_batches = n_batches

        self.model.train()
        for batch_number, (images_A, images_B, pair_indices) in enumerate(
            self.train_loader, starting_batch + 1
        ):
            embedded_A = self.model.forward(images_A.to(DEVICE, non_blocking=True))
            embedded_B = self.model.forward(images_B.to(DEVICE, non_blocking=True))
            self.optimizer.zero_grad(set_to_none=True)
            positive_pairs = pair_indices >= self.n_negative_pairs
            losses = self.criterion(embedded_A, embedded_B, positive_pairs)
            losses.mean().backward()
            self.optimizer.step()

            positive_losses = losses[positive_pairs]
            negative_losses = losses[~positive_pairs]

            n_positive = len(positive_losses)
            n_negative = len(negative_losses)
            n_loss_positive = positive_losses.count_nonzero().item()
            n_loss_negative = negative_losses.count_nonzero().item()

            self.positive_err_rate += n_loss_positive / max(n_positive, 1)
            self.negative_err_rate += n_loss_negative / max(n_negative, 1)
            self.penalties += pair_indices.bincount(
                (losses != 0).detach().cpu(), minlength=len(self.penalties)
            )

            self.positive_err_rate *= 0.98
            self.negative_err_rate *= 0.98
            self.penalties *= 0.98

            self.train_loader.batch_sampler.weights = get_weights(  # type: ignore
                self.negative_weights,
                self.positive_weights,
                self.negative_penalties,
                self.positive_penalties,
                self.positive_err_rate,
                self.negative_err_rate,
            )

            status.update(
                f"[red]Batch {batch_number:2}: sampled {n_positive} positive pairs ({n_loss_positive} with loss) and "
                f"{n_negative} negative pairs ({n_loss_negative} with loss)"
            )

    @torch.inference_mode()
    def predict(self, fragments: ListOfFragments) -> None:
        image_locations: list[tuple[int, int]] = []
        lengths: list[int] = []
        candidate_fragments_identifiers: list[int] = []

        for fragment in fragments.individual_fragments:
            image_locations += fragment.image_locations
            lengths.append(fragment.n_images)
            candidate_fragments_identifiers.append(fragment.identifier)

        assert image_locations

        logging.debug(
            "Predicting %d images with contrastive model", len(image_locations)
        )

        dataloader = get_onthefly_dataloader(
            image_locations, fragments.id_images_file_paths
        )

        self.model.eval()
        embeddings = np.concatenate([
            self.model.forward(images.to(DEVICE) / 255).numpy(force=True)
            for images, _labels in track(dataloader, "Predicting")
        ])

        kmeans = MiniBatchKMeans(self.n_animals, n_init=50).fit(embeddings)
        distances = kmeans.transform(embeddings)

        prob: np.ndarray = np.reciprocal(distances + 0.01) ** 7
        prob /= prob.sum(1, keepdims=True)

        assignments: np.ndarray = prob.argmax(1, keepdims=True)
        probabilities = np.take_along_axis(prob, assignments, axis=1)

        self.cluter_centers = kmeans.cluster_centers_

        logging.debug("Computing fragment prediction statistics")

        fragments_assignments = np.split(
            assignments.flatten() + 1, np.cumsum(lengths)[:-1]
        )
        fragments_probabilities = np.split(
            probabilities.flatten(), np.cumsum(lengths)[:-1]
        )

        for predictions, probabilities, fragment in zip(
            fragments_assignments,
            fragments_probabilities,
            fragments.individual_fragments,
        ):
            fragment.compute_identification_statistics(
                predictions, probabilities, self.n_animals
            )


def get_weights(
    negative_weights: Tensor,
    positive_weights: Tensor,
    negative_scores: Tensor,
    positive_scores: Tensor,
    positive_err_rate: float = 1,
    negative_err_rate: float = 1,
) -> Tensor:
    sum_err_rates = positive_err_rate + negative_err_rate
    return torch.concatenate((
        max(positive_err_rate, 0.05 * sum_err_rates)
        * (negative_weights + negative_scores / negative_scores.sum()),
        max(negative_err_rate, 0.05 * sum_err_rates)
        * (positive_weights + positive_scores / positive_scores.sum()),
    ))


def val_collate_fun(
    batch: list[tuple[Tensor]],
    id_images_paths: Sequence[Path],
    loaded_images: list[np.ndarray] | None = None,
) -> list[Tensor]:
    """Receives the batch images locations (episode and index).
    These are used to load the images and generate the batch tensor"""
    locations = torch.stack(tuple(zip(*batch))[0]).numpy()

    if loaded_images is None:
        # there are no preloaded images, lets get them from disk
        images = load_id_images(
            id_images_paths, locations, verbose=False, dtype=np.float32
        )
    else:
        # images are in RAM
        img_indices, episodes = np.asarray(locations).T
        images = np.empty((len(img_indices), *loaded_images[0].shape[1:]), np.float32)

        for episode in np.unique(episodes):
            where = episodes == episode
            images[where] = loaded_images[episode][img_indices[where]]

    return [torch.from_numpy(images).contiguous().unsqueeze(1) / 255]


def collate_fun(
    batch: list[tuple[tuple[int, int], tuple[int, int], int]],
    id_images_paths: Sequence[Path],
    loaded_images: list[np.ndarray] | None = None,
) -> list[Tensor]:
    """Receives the batch images locations (episode and index).
    These are used to load the images and generate the batch tensor"""
    locations_A, locations_B, labels = zip(*batch)

    if loaded_images is None:
        # there are no preloaded images, lets get them from disk
        images = load_id_images(
            id_images_paths, locations_A + locations_B, verbose=False, dtype=np.float32
        )
    else:
        # images are in RAM
        img_indices, episodes = np.asarray(locations_A + locations_B).T
        images = np.empty((len(img_indices), *loaded_images[0].shape[1:]), np.float32)

        for episode in np.unique(episodes):
            where = episodes == episode
            images[where] = loaded_images[episode][img_indices[where]]

    images = torch.from_numpy(images).contiguous().unsqueeze(1)
    images /= 255
    return [
        images[: len(locations_A)],
        images[len(locations_A) :],
        torch.tensor(labels),
    ]
