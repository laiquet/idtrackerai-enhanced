"Contrastive Learning module"

import logging
import random
from dataclasses import dataclass
from functools import partial
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Iterator, Protocol, Sequence

import numpy as np
import torch
from h5py import File
from rich.console import Console
from rich.status import Status
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
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
        self, weights: Tensor, batch_size: int, n_batches: int = 2**100
    ) -> None:
        self.weights = weights
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.n_batches):
            yield torch.multinomial(
                self.weights, self.batch_size, replacement=True
            ).tolist()

    def __len__(self) -> int:
        return self.n_batches


class ContrastiveDataLoader(Protocol):
    """Protocol class for better typing our DataLoaders"""

    batch_sampler: BatchSampler
    dataset: PairsOfFragments

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]: ...


@dataclass(slots=True)
class ContrastiveLearning:
    model: ResNet
    optimizer: torch.optim.Optimizer
    loaded_images: list[np.ndarray] | None
    val_loader: ContrastiveDataLoader
    train_loader: ContrastiveDataLoader
    scores: Tensor
    positive_err_rate: float
    negative_err_rate: float

    negative_weights: Tensor
    positive_weights: Tensor

    n_negative_pairs: int
    check_every: int

    n_animals: int

    first_batch_group_to_check: int
    learning_rate: float
    embedding_dimensions: int

    cluter_centers: np.ndarray

    @property
    def negative_scores(self) -> Tensor:
        return self.scores[: self.n_negative_pairs]

    @property
    def positive_scores(self) -> Tensor:
        return self.scores[self.n_negative_pairs :]

    def __init__(
        self,
        fragments: ListOfFragments,
        check_every: int = 1000,
        batch_size: int = 500,
        preload_images: bool = False,
        min_frag_length: int = 4,
        learning_rate: float = 0.001,
        embedding_dimensions: int = 8,
        first_batch_group_to_check: int = 3,
    ) -> None:
        self.first_batch_group_to_check = first_batch_group_to_check
        self.learning_rate = learning_rate
        self.embedding_dimensions = embedding_dimensions
        self.check_every = check_every
        self.n_animals = fragments.n_animals

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
        self.scores = torch.full([len(pairs_of_fragments)], 10, dtype=torch.double)

        if preload_images:
            self.loaded_images: list[np.ndarray] | None = []
            for path in fragments.id_images_file_paths:
                with File(path) as file:
                    self.loaded_images.append(file["id_images"][:])  # type: ignore
        else:
            self.loaded_images = None

        self.build_dataloaders(
            pairs_of_fragments, batch_size, fragments.id_images_file_paths
        )

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
        batch_size: int,
        id_images_file_paths: Sequence[Path],
    ) -> None:

        dataset = PairsOfFragments(pairs_of_fragments)

        collate_fn = partial(
            collate_fun,
            id_images_paths=id_images_file_paths,
            loaded_images=self.loaded_images,
        )

        # if images are not loaded, we need more workers to load them on the fly
        num_workers = 6 if self.loaded_images is None else 3

        self.val_loader = DataLoader(  # type:ignore
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=BatchSampler(
                weights=get_weights(
                    self.negative_weights,
                    self.positive_weights,
                    self.positive_scores * 0 + 1,
                    self.negative_scores * 0 + 1,
                ),
                batch_size=batch_size,
                n_batches=int(10_000 / batch_size) + 1,
            ),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.train_loader = DataLoader(  # type:ignore
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=BatchSampler(
                weights=get_weights(
                    self.negative_weights,
                    self.positive_weights,
                    self.positive_scores,
                    self.negative_scores,
                ),
                batch_size=batch_size,
            ),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def train(self, reset_model: bool = True) -> None:
        "Main method to train the contrastive"
        if reset_model:
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

        self.model.train()
        start = perf_counter()
        with Console().status("Training contrastive") as status:
            for batch_group in count():
                start = perf_counter()

                self.train_step(
                    status,
                    n_batches=self.check_every,
                    starting_batch=batch_group * self.check_every,
                )
                stop = perf_counter()

                if batch_group < self.first_batch_group_to_check:
                    continue

                status.update("Validating")

                distance = self.validate()

                logging.debug(
                    f"Batch: {batch_group*self.check_every}-{(batch_group+1)*self.check_every} {self.check_every/(stop-start):5.1f} batches/s | 90% distance percentile = {distance:.2f} (stop training when < 1)"
                )
                if distance < 1:
                    break

    @torch.inference_mode()
    def validate(self) -> np.float_:
        "Clustering images from self.val_loader and return the 90% percentile of the distance to the closest cluster."
        self.model.eval()
        embeddings = []
        for images_A, images_B, _pair_indices in self.val_loader:
            embeddings += (
                self.model.forward(images_A.to(DEVICE, non_blocking=True)).numpy(
                    force=True
                ),
                self.model.forward(images_B.to(DEVICE, non_blocking=True)).numpy(
                    force=True
                ),
            )

        distances = MiniBatchKMeans(self.n_animals, n_init=50).fit_transform(
            np.concatenate(embeddings)
        )

        # assign closest cluster to every image and take this distance
        cluster_labels = distances.argmin(1)
        cluster_distances = np.take_along_axis(
            distances, cluster_labels[:, None], axis=1
        )

        return np.percentile(cluster_distances, 90)

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

            self.positive_err_rate += n_loss_positive / n_positive
            self.negative_err_rate += n_loss_negative / n_negative
            self.scores += pair_indices.bincount(
                (losses != 0).detach().cpu(), minlength=len(self.scores)
            )

            self.positive_err_rate *= 0.98
            self.negative_err_rate *= 0.98
            self.scores *= 0.98

            self.train_loader.batch_sampler.weights = get_weights(  # type: ignore
                self.negative_weights,
                self.positive_weights,
                self.positive_scores,
                self.negative_scores,
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
            self.model.forward(images.to(DEVICE)).numpy(force=True)
            for images, _labels in track(dataloader, "Predicting")
        ])

        kmeans = MiniBatchKMeans(self.n_animals, n_init=50).fit(embeddings)
        self.cluter_centers = kmeans.cluster_centers_
        predictions = kmeans.labels_

        assert sum(lengths) == len(predictions)

        logging.debug("Computing fragment prediction statistics")

        fragments_predictions = np.split(predictions, np.cumsum(lengths)[:-1])

        for predictions, fragment in zip(
            fragments_predictions, fragments.individual_fragments
        ):
            fragment.compute_identification_statistics(
                predictions, None, self.n_animals
            )


def get_weights(
    negative_weights: Tensor,
    positive_weights: Tensor,
    positive_scores: Tensor,
    negative_scores: Tensor,
    mean_negative_loss: float = 1,
    mean_positive_loss: float = 1,
) -> Tensor:
    return torch.concatenate((
        mean_negative_loss
        * (negative_weights + negative_scores / negative_scores.sum()),
        mean_positive_loss
        * (positive_weights + positive_scores / positive_scores.sum()),
    ))


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
