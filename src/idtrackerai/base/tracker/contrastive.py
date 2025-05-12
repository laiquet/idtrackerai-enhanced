"Contrastive Learning module"

import json
import logging
import random
import signal
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial, wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import h5py
import numpy as np
import psutil
import torch
from rich.console import Console
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from torch import Tensor
from torch.nn.functional import pairwise_distance, relu
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

from idtrackerai import (
    Fragment,
    GlobalFragment,
    IdtrackeraiError,
    ListOfFragments,
    __version__,
    conf,
)
from idtrackerai.base.network import (
    DEVICE,
    IdentifierContrastive,
    ResNet18,
    get_onthefly_dataloader,
)
from idtrackerai.utils import H5DatasetProxy, load_id_images, track


def _ignore_sigint_in_worker(_worker_id: int) -> None:
    """
    Function to ignore SIGINT signal in workers.

    This is used to allow user to stop training with a KeyboardInterrupt.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PairsOfFragments(Dataset):
    """
    Dataset with all pairs of fragments (positive and negative) which returns
    only the locations of selected images (the images are loaded in collate_fun).

    Attributes:
        pairs (list[tuple[Fragment, Fragment]]): List of fragment pairs.
    """

    pairs: list[tuple[Fragment, Fragment]]

    def __init__(self, pairs: list[tuple[Fragment, Fragment]]) -> None:
        super().__init__()
        self.pairs = pairs

    def __len__(self) -> int:
        """
        Return the number of pairs in the dataset.
        """
        return len(self.pairs)

    def __getitem__(
        self, pair_index: int
    ) -> tuple[tuple[int, int], tuple[int, int], int]:
        """
        Get two images from a specific pair of fragments.

        Parameters:
            pair_index (int): Index of the pair to retrieve.

        Returns:
            tuple: A tuple containing:
                - Locations of an image from the first fragment.
                - Locations of an image from the second fragment.
                - Index of the pair.
        """
        frag_A, frag_B = self.pairs[pair_index]
        img_index_A = random.randint(0, frag_A.n_images - 1)
        img_index_B = random.randint(0, frag_B.n_images - 1)

        return (
            (frag_A.images[img_index_A], frag_A.episodes[img_index_A]),
            (frag_B.images[img_index_B], frag_B.episodes[img_index_B]),
            pair_index,
        )


class BatchSampler(Sampler[list[int]]):
    """
    Custom implementation of a torch.utils.data.BatchSampler.

    This sampler generates batches of indices based on a probability distribution
    derived from loss scores and pair sizes. The probabilities can be updated
    dynamically during training.

    Attributes:
        batch_size (int): Number of samples in each batch.
        n_batches (int | None): Number of batches to generate.
        negative_pairs_sizes (Tensor): Sizes of negative pairs.
        positive_pairs_sizes (Tensor): Sizes of positive pairs.
        negative_probabilities (Tensor): Sampling probabilities for negative pairs.
        positive_probabilities (Tensor): Sampling probabilities for positive pairs.
    """

    def __init__(
        self,
        negative_pairs_sizes: Tensor,
        positive_pairs_sizes: Tensor,
        negative_loss_scores: Tensor,
        positive_loss_scores: Tensor,
        batch_size: int,
        n_batches: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.negative_pairs_sizes = negative_pairs_sizes
        self.positive_pairs_sizes = positive_pairs_sizes
        self.update_probabilities(negative_loss_scores, positive_loss_scores)

    def __iter__(self) -> Iterator[list[int]]:
        """
        Generate batches of indices.

        Yields:
            list[int]: A batch of indices.
        """
        if self.n_batches is None:
            raise RuntimeError(
                "BatchSampler has n_batches set to None. Please set n_batches before iterating"
            )
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
        self, negative_scores: Tensor, positive_scores: Tensor
    ) -> None:
        """
        Update the sampling probabilities for negative and positive pairs based on their
        loss scores. The parameter ``\alpha`` in the 2025 article is hardcoded to 0.5 so
        both probability contributions are summed with the same weight.

        Parameters:
            negative_scores (Tensor): Loss scores for negative pairs.
            positive_scores (Tensor): Loss scores for positive pairs.
        """
        self.negative_probabilities = (
            self.negative_pairs_sizes + negative_scores / negative_scores.sum()
        )
        self.positive_probabilities = (
            self.positive_pairs_sizes + positive_scores / positive_scores.sum()
        )


class ContrastiveDataLoader(Protocol):
    """Protocol class for better typing our DataLoaders"""

    batch_sampler: BatchSampler
    dataset: PairsOfFragments

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]: ...


def _catch_out_of_memory(function: Callable):
    """Decorator to catch OutOfMemory errors and raise a more informative error"""

    @wraps(function)
    def wrapped_function(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            raise IdtrackeraiError(
                'GPU got out of memory. Decrease the "CONTRASTIVE_BATCHSIZE" parameter, '
                f"current value is {conf.CONTRASTIVE_BATCHSIZE}"
            ) from exc

    return wrapped_function


class ContrastiveLearning:
    """Main API class for Contrastive Learning."""

    model: ResNet18
    """RasNet18 model"""
    optimizer: Optimizer
    """Optimizer"""
    val_loader: ContrastiveDataLoader
    """Validation DataLoader"""
    train_loader: ContrastiveDataLoader
    """Contrastive training DataLoader"""
    gfrag_loader: ContrastiveDataLoader | None
    """DataLoader for a single Global Fragments images used to initialize kmeans clusters.
    None if there are no Global Fragments in the video."""
    cluster_centers: np.ndarray
    loss_scores: Tensor
    """Sequence of floats representing the loss scores of every pair of Fragments used in contrastive.
    Loss scores increase when a pair of images is sampled from a specific pair of Fragments and its loss is non zero."""
    n_negative_pairs: int
    """The number of negative pairs of Fragments we have"""
    n_animals: int
    """Number of animals in the video"""
    learning_rate: float
    """Optimizer learning rate"""
    embedding_dimensions: int
    """Number of dimensions of the embedded space"""
    saving_folder: Path
    """Saving folder for checkpoints"""
    batch_size: int
    """Number of pairs of each kind of images (positive and negative) used in a single training batch"""
    checkpoint_filename: str = "contrastive_checkpoint.pt"

    @property
    def negative_loss_scores(self) -> Tensor:
        return self.loss_scores[: self.n_negative_pairs]

    @property
    def positive_loss_scores(self) -> Tensor:
        return self.loss_scores[self.n_negative_pairs :]

    @property
    def model_checkpoint_path(self) -> Path:
        return self.saving_folder / self.checkpoint_filename

    def __init__(
        self,
        fragments: ListOfFragments,
        saving_folder: Path | None = None,
        first_gfrag: GlobalFragment | None = None,
        batch_size: int = conf.CONTRASTIVE_BATCHSIZE,
        preload_images_max_mbytes: float | None = conf.CONTRASTIVE_MAX_MBYTES,
        learning_rate: float = 0.001,
        embedding_dimensions: int = 8,
        min_fragment_length: int = conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION,
    ) -> None:
        if saving_folder is not None:
            self.saving_folder = saving_folder
        self.learning_rate = learning_rate
        self.embedding_dimensions = embedding_dimensions
        self.n_animals = fragments.n_animals
        self.batch_size = batch_size

        fragments_selection = [
            frag
            for frag in fragments
            if frag.is_an_individual and frag.n_images >= min_fragment_length
        ]
        logging.info(
            f"Out of {len(fragments.fragments)} fragments, {len(fragments_selection)} "
            f"are individuals and longer than {min_fragment_length - 1} frames, they are gonna be used for contrastive training"
        )

        pairs_of_fragments: list[tuple[Fragment, Fragment]] = []
        for fragment in fragments_selection:
            for coex_frag in fragment.coexisting_individual_fragments:
                if (
                    coex_frag.identifier > fragment.identifier
                    and coex_frag.is_an_individual
                    and coex_frag.n_images >= min_fragment_length
                ):
                    pairs_of_fragments.append((fragment, coex_frag))

        self.n_negative_pairs = len(pairs_of_fragments)
        pairs_of_fragments += ((frag, frag) for frag in fragments_selection)
        logging.info(
            f"Generated {self.n_negative_pairs} negative and "
            f"{len(pairs_of_fragments) - self.n_negative_pairs} positive pairs of Fragments"
        )

        if not self.n_negative_pairs:
            raise IdtrackeraiError(
                "There are no negative pairs of Fragments to train Contrastive. "
                "Idtracker.ai requires pairs of animals to appear together in the "
                "video for contrastive to lear to distinguish between them."
            )

        if self.n_negative_pairs < 0.25 * (
            len(fragments_selection) * (self.n_animals - 1)
        ):
            logging.warning(
                "The animals in the video appear too isolated. idtracker.ai relies on observing groups of animals visible at the same time to effectively train the model. Limited coexistence may reduce tracking accuracy."
            )

        self.loss_scores = torch.full([len(pairs_of_fragments)], 10, dtype=torch.double)

        # Identification images sources. Can be Numpy arrays preloaded in memory or HDF5 Datasets views to disk
        image_sources = self.preload_images(
            fragments.id_images_file_paths, preload_images_max_mbytes
        )
        self.build_dataloaders(
            pairs_of_fragments,
            fragments_selection,
            image_sources,
            1000 * self.n_animals,
            first_gfrag,
        )

    @staticmethod
    def preload_images(
        paths: Iterable[Path], max_memory_usage: float | None = None
    ) -> list[h5py.Dataset] | list[np.ndarray] | list[H5DatasetProxy]:
        if max_memory_usage is None:
            max_memory_usage = psutil.virtual_memory().available / (4 * 1024**2)
            logging.info(
                f"Maximum memory usage for pre-loading images not set (CONTRASTIVE_MAX_MBYTES=None), using a forth of the available memory in the system: {max_memory_usage:.1f} MB"
            )

        n_megabytes = sum(
            h5py.File(path)["id_images"].nbytes  # type:ignore
            for path in paths
        ) / (1024 * 1024)
        logging.info(
            f"All identification images weight {n_megabytes:.1f} MB. The stated limit for them to be pre-loaded is {max_memory_usage:.1f} MB"
        )

        if n_megabytes > max_memory_usage:
            logging.info(
                "Not pre-loading identification images, they will be loaded from disk on demand"
            )
            return [H5DatasetProxy(path, "id_images") for path in paths]

        return [  # type:ignore
            h5py.File(path)["id_images"][:]  # type:ignore
            for path in track(paths, "Pre-loading all identification images to RAM")
        ]

    @staticmethod
    def criterion(
        embedded_A: Tensor, embedded_B: Tensor, first_positive: int, margin: float = 10
    ) -> Tensor:
        """Pairwise distance loss criterion.
        Negative pairs are pushed away until they are at distance `margin`.
        Positive pairs are pulled together until they are at distance 1"""
        # TODO implement torch.compile
        distance = pairwise_distance(embedded_A, embedded_B)

        losses = torch.concatenate(  # negative first, positive after
            (margin - distance[:first_positive], distance[first_positive:] - 1)
        )
        return relu(losses).square()

    def build_dataloaders(
        self,
        pairs_of_fragments: list[tuple[Fragment, Fragment]],
        fragments_selection: Iterable[Fragment],
        image_sources: Sequence[np.ndarray | h5py.Dataset | H5DatasetProxy],
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
            torch.tensor(val_images), torch.zeros(len(val_images), dtype=torch.int8)
        )  # dummy light-weight labels to reuse dataloader code

        collate_fn = partial(collate_fun, images_sources=image_sources)

        if not isinstance(image_sources[0], np.ndarray):
            # if images are not loaded, we need more workers to load them on the fly
            num_workers = 3
        else:
            # if we already have preloaded images, we don't want many workers
            num_workers = 1

        logging.info(
            f"Using {num_workers} workers to load images for the training and validation DataLoaders"
        )

        negative_pairs_sizes = torch.tensor(
            [
                frag.n_images + coex_frag.n_images
                for frag, coex_frag in pairs_of_fragments[: self.n_negative_pairs]
            ],
            dtype=torch.float64,
        )
        positive_pairs_sizes = torch.tensor(
            [
                frag.n_images
                for frag, same_frag in pairs_of_fragments[self.n_negative_pairs :]
            ],
            dtype=torch.float64,
        )
        negative_pairs_sizes /= negative_pairs_sizes.sum()
        positive_pairs_sizes /= positive_pairs_sizes.sum()

        self.val_loader = DataLoader(  # type:ignore
            dataset=val_dataset,
            num_workers=num_workers,
            batch_size=self.batch_size,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=_ignore_sigint_in_worker,
        )

        self.train_loader = DataLoader(  # type:ignore
            dataset=train_dataset,
            num_workers=num_workers,
            batch_sampler=BatchSampler(
                negative_pairs_sizes,
                positive_pairs_sizes,
                self.negative_loss_scores,
                self.positive_loss_scores,
                self.batch_size,
            ),
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=_ignore_sigint_in_worker,
        )

        if first_gfrag is None:
            logging.info(
                'Using "k-means++" as K-Means clustering initializer because '
                "there are no Global Fragments"
            )
            self.gfrag_loader = None
            return

        if first_gfrag.min_n_images_per_fragment < 30:
            logging.info(
                'Using "k-means++" as K-Means clustering initializer because '
                f"the biggest Global Fragment ({first_gfrag.min_n_images_per_fragment}"
                " frames) is not big enough (30 frames)"
            )
            self.gfrag_loader = None
            return

        image_locations = []
        frag_ids = []
        for frag_id, fragment in enumerate(first_gfrag):
            if fragment.n_images > 500:
                # we do not need more than 500 images per animal to initialize KMeans
                image_locations += random.sample(list(fragment.image_locations), 500)
                frag_ids += [frag_id] * 500
            else:
                image_locations += fragment.image_locations
                frag_ids += [frag_id] * fragment.n_images

        logging.info(
            f"Using {len(image_locations)} images from the global"
            f" fragment starting at frame {first_gfrag.first_frame_of_the_core} as"
            " the groundtruth dataset to initialize K-Means clustering"
        )

        first_gfrag_images = (
            torch.from_numpy(
                load_id_images(image_sources, image_locations, False, np.float32)
            )
            .contiguous()
            .unsqueeze(1)
            / 255
        )
        first_gfrag_dataset = TensorDataset(first_gfrag_images, torch.tensor(frag_ids))

        self.gfrag_loader = DataLoader(  # type:ignore
            first_gfrag_dataset, batch_size=self.batch_size
        )

    def set_model(
        self, weights_path: Path | None = None, compile: bool = conf.TORCH_COMPILE
    ) -> None:
        "Initializes the contrastive model from a knowledge transfer file or from scratch"

        if weights_path is not None:
            # initialize model with knowledge transfer
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Knowledge transfer path {weights_path} not found"
                )
            elif weights_path.is_file():
                pass
            elif (
                weights_path / IdentifierContrastive.model_weights_filename
            ).is_file():
                # We found the Identifier model!
                weights_path /= IdentifierContrastive.model_weights_filename
            elif (weights_path / self.checkpoint_filename).is_file():
                # there is not an Identifier model but we there's a checkpoint, better than nothing!
                weights_path /= self.checkpoint_filename
            else:
                raise FileNotFoundError(
                    f"Could not find a Contrastive model weights in {weights_path}"
                )

            self.model = ResNet18.from_file(weights_path)
        else:
            # initialize model from scratch
            logging.info("Randomly initializing contrastive model")
            self.model = ResNet18(
                n_channels_in=1, n_dimensions_out=self.embedding_dimensions
            )

        self.model = self.model.to(DEVICE)
        if compile:
            self.model.compile()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def train_step(
        self,
        n_batches: int,
        output: Callable[[str], None] = lambda x: None,
        starting_batch_number: int = 0,
    ) -> tuple[float, float]:
        """
        Perform n_batches training steps.

        Parameters:
            n_batches (int): Number of batches to train on.
            output (Callable[[str], None]): Function to output training progress.
            starting_batch_number (int): Starting batch number for logging.

        Returns:
            tuple[float, float]: Fraction of positive and negative sampled pairs with loss.
        """
        # this will make the dataloader to iterate n_batches times
        self.train_loader.batch_sampler.n_batches = n_batches

        try:
            self.model.train()
        except AttributeError:
            raise IdtrackeraiError(
                "Call ContrastiveLearning.set_model() before training"
            )

        total_n_loss_positive = 0
        total_n_loss_negative = 0
        n_pairs = 0
        for batch_number, (images_A, images_B, pair_indices) in enumerate(
            self.train_loader, starting_batch_number + 1
        ):
            # each batch has batch_size negative pairs with
            #     images_A[: self.batch_size] -> images_B[: self.batch_size]
            # and batch_size positive pairs with
            #     images_A[self.batch_size:] -> images_B[self.batch_size:]
            # So, in each batch ResNet sees 4*batch_size images

            images_A = images_A.to(DEVICE, non_blocking=True)
            images_B = images_B.to(DEVICE, non_blocking=True)
            embedded_A = self.model(images_A)
            embedded_B = self.model(images_B)
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.criterion(embedded_A, embedded_B, self.batch_size)
            losses.mean().backward()
            self.optimizer.step()

            has_loss = (losses.detach() != 0).cpu()
            n_loss_negative = losses[: self.batch_size].count_nonzero().item()
            n_loss_positive = losses[self.batch_size :].count_nonzero().item()

            self.loss_scores += pair_indices.bincount(
                has_loss, minlength=len(self.loss_scores)
            )

            self.loss_scores *= 0.98

            self.train_loader.batch_sampler.update_probabilities(
                self.negative_loss_scores, self.positive_loss_scores
            )

            total_n_loss_positive += n_loss_positive
            total_n_loss_negative += n_loss_negative
            n_pairs += self.batch_size

            output(
                f"[red]Batch {batch_number:2}: sampled {self.batch_size} positive pairs ({n_loss_positive:3d} "
                f"too far apart) and {self.batch_size} negative pairs ({n_loss_negative:3d} too close)"
            )
        return total_n_loss_positive / n_pairs, total_n_loss_negative / n_pairs

    @_catch_out_of_memory
    def train(
        self,
        check_every: int | None = None,
        skipped_validations: int = 5,
        patience: int = conf.CONTRASTIVE_PATIENCE,
        target_silhouette_score: float = conf.CONTRASTIVE_SILHOUETTE_TARGET,
    ) -> float:
        "Main method to train the contrastive"
        if not hasattr(self, "saving_folder"):
            raise IdtrackeraiError(
                "The saving_folder was not set at ContrastiveLearning initialization. Please set it before training"
            )

        if not check_every:
            check_every = max(5 * self.n_animals, 100)
        first_batch_to_validate = skipped_validations * check_every

        (self.saving_folder / "model_params.json").write_text(
            json.dumps(
                {
                    "n_classes": self.n_animals,
                    "model": self.model.__class__.__name__,
                    "version": __version__,
                },
                indent=4,
            )
        )
        best_score: float = 0
        steps_without_improvement: int = 0
        batch_counter: int = 0
        logging.debug(
            "[bold]Batch | Batches/s | Silhouette score | Too far apart positive pairs | Too close negative pairs",
            extra={"markup": True},
        )
        backup_gfrag_loader = None
        best_score_used_gfrag: bool | None = None

        try:
            with Console().status("Initializing contrastive training") as status:
                while True:
                    start = perf_counter()
                    positive_losses, negative_losses = self.train_step(
                        n_batches=check_every,
                        output=status.update,
                        starting_batch_number=batch_counter,
                    )
                    batch_counter += check_every
                    stop = perf_counter()

                    if batch_counter < first_batch_to_validate:
                        continue

                    status.update("Validating")

                    silhouette_score = self.validate()

                    status.stop()
                    logging.debug(
                        f"{batch_counter:5d} |{check_every / (stop - start):7.1f}"
                        f"    |      {silhouette_score:6.4f}"
                        f"{'!' if silhouette_score > best_score else '':<6}|"
                        f"{positive_losses:>18.1%}{' |':^24}{negative_losses:5.1%}"
                    )
                    status.start()

                    if silhouette_score > best_score:
                        if best_score < target_silhouette_score < silhouette_score:
                            logging.info(
                                "[bold]The silhouette score of CONTRASTIVE_SILHOUETTE_TARGET="
                                f"{target_silhouette_score} [green]has been achieved![/][/]\n"
                                "We will stop the training now after 2 steps without improvements",
                                extra={"markup": True},
                            )
                        best_score = silhouette_score
                        best_score_used_gfrag = self.gfrag_loader is not None
                        torch.save(self.model.state_dict(), self.model_checkpoint_path)
                        steps_without_improvement = 0
                    else:
                        steps_without_improvement += 1

                    if steps_without_improvement > patience:
                        if self.gfrag_loader is not None:
                            logging.info(
                                f"The model has not improved for CONTRASTIVE_PATIENCE={patience} steps. "
                                'Before stop training, we will try to initialize KMeans with "k-means++" '
                                "rather than using the images from a Global Fragment"
                            )
                            backup_gfrag_loader = self.gfrag_loader
                            self.gfrag_loader = None
                            steps_without_improvement -= 5
                            continue

                        logging.warning(
                            f"The model has not improved for CONTRASTIVE_PATIENCE={patience}"
                            " steps, we stop the training. Increase the value of this parameter "
                            "to allow further training."
                        )
                        break

                    if (
                        best_score > target_silhouette_score
                        and steps_without_improvement > 1
                    ):
                        logging.info(
                            "The model has not improved for 2 steps, but the target "
                            f"silhouette score ({target_silhouette_score}) was already achieved"
                        )
                        break

                    if batch_counter > 100_000 * check_every:
                        # This should never happen, but just in case
                        logging.warning("Maximum number of training batches reached")
                        break
        except KeyboardInterrupt:
            logging.warning("Training interrupted by user")

        if self.model_checkpoint_path.is_file():
            logging.info(
                "Loading best model weights from the checkpoint with silhouette score %.7f",
                best_score,
            )
            self.model.load_state_dict(
                torch.load(self.model_checkpoint_path, weights_only=True)
            )
            if (
                best_score_used_gfrag is True
                and self.gfrag_loader is None
                and backup_gfrag_loader is not None
            ):
                self.gfrag_loader = backup_gfrag_loader
                logging.info(
                    "Restoring the Global Fragment KMeans initialization since its removal did not improve results"
                )
        return best_score

    @torch.inference_mode()
    def validate(self) -> float:
        """
        Validate the model using :func:`silhouette_scores`.

        Returns:
            float: Average silhouette score.

        TODO: Check a possible improvement of the clustering method with
        https://contrib.scikit-learn.org/metric-learn/generated/metric_learn.LMNN.html
        """
        self.model.eval()
        embeddings = torch.concatenate(
            [self.model(images.to(DEVICE)) for (images, _labels) in self.val_loader]
        ).detach()
        kmeans = MiniBatchKMeans(self.n_animals, **self.kmeans_init())
        labels = torch.from_numpy(kmeans.fit_predict(embeddings.numpy(force=True)))
        return silhouette_scores(embeddings, labels.to(DEVICE)).mean().item()

    @torch.inference_mode()
    def predict(
        self, fragments: ListOfFragments, reference_gfrag: GlobalFragment | None = None
    ) -> None:
        """
        Predict cluster assignments for fragments.

        Parameters:
            fragments (ListOfFragments): List of fragments to predict.
            reference_gfrag (GlobalFragment | None): Reference Global Fragment for identity mapping.
        """
        "Returns the mean silhouette score per identity"
        image_locations: list[tuple[int, int]] = []
        lengths: list[int] = []

        # we will predict with all fragments (individual and long enough)
        # even if none is accumulable (no global fragments) because this
        # prediction will be used to compute the cluster centers for the
        # downstream processing (basically residual identification)
        frags_to_predict = [
            frag
            for frag in fragments
            if frag.is_an_individual
            and frag.n_images >= conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
        ]

        for fragment in frags_to_predict:
            image_locations += fragment.image_locations
            lengths.append(fragment.n_images)
        sections = np.cumsum(lengths)[:-1]  # used to de-flatten the predictions arrays

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
                self.model(images.to(DEVICE)).numpy(force=True)
                for images, _labels in track(dataloader, "Predicting")
            ]
        )

        kmeans = MiniBatchKMeans(self.n_animals, **self.kmeans_init())
        distances = kmeans.fit_transform(embeddings)
        # These will be the cluster centers used in all downstream processes
        self.cluster_centers = kmeans.cluster_centers_

        prob: np.ndarray = np.reciprocal(distances + 0.01) ** 7
        prob /= prob.sum(1, keepdims=True)

        assignments: np.ndarray = prob.argmax(1)
        probabilities = prob[range(len(prob)), assignments]

        fragments_assignments = np.split(assignments + 1, sections)
        fragments_probabilities = np.split(probabilities, sections)

        if reference_gfrag is not None:
            # if there is a first Global Fragment, it should be already assigned with "temporary_id" from identity transfer or exclusive ROIs...
            # We will adapt cluster assignments to these identities
            translation_matrix = np.empty((self.n_animals, self.n_animals), int)
            for frag in reference_gfrag:
                assert frag.temporary_id is not None
                translation_matrix[frag.temporary_id] = np.bincount(
                    fragments_assignments[frags_to_predict.index(frag)] - 1,
                    minlength=self.n_animals,
                )
            ids_map = linear_sum_assignment(translation_matrix.T, maximize=True)[1]

            if not np.array_equal(ids_map, np.arange(self.n_animals)):
                logging.info(
                    "Using the identities from the reference Global Fragment to reorder contrastive clusters"
                )
                reorder = np.argsort(ids_map)
                self.cluster_centers = self.cluster_centers[reorder]
                assignments = np.vectorize(lambda x: ids_map[x])(assignments)
                fragments_assignments = np.split(assignments + 1, sections)

        for predictions, probabilities, fragment in zip(
            track(fragments_assignments, "Computing fragment prediction statistics"),
            fragments_probabilities,
            frags_to_predict,
        ):
            fragment.set_identification_statistics(
                predictions, probabilities, self.n_animals
            )

    def get_identification_model(self) -> IdentifierContrastive:
        return IdentifierContrastive(
            model=self.model, cluster_centers=torch.from_numpy(self.cluster_centers)
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
            embeddings.append(self.model(images.to(DEVICE)).numpy(force=True))
            labels.append(labels_.numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        cluster_centers = np.asarray(
            [embeddings[labels == label].mean(0) for label in range(self.n_animals)]
        )
        return {"batch_size": batch_size, "n_init": 1, "init": cluster_centers}


def collate_fun(
    batch: list[tuple[tuple[int, int], tuple[int, int], int]],
    images_sources: Sequence[h5py.Dataset | Path | str | np.ndarray | H5DatasetProxy],
) -> list[Tensor]:
    """Receives the batch with N groups of images locations (episode and index) and a label.
    These are used to load the images and generate the batch tensor"""
    unzziped_batch = list(zip(*batch))
    locations, labels = unzziped_batch[:-1], unzziped_batch[-1]
    images = load_id_images(
        images_sources, np.concatenate(locations), verbose=False, dtype=np.float32
    )
    images = torch.from_numpy(images).contiguous().unsqueeze(1) / 255
    return [
        *torch.split(images, [len(location) for location in locations]),
        torch.tensor(labels),
    ]


def silhouette_scores(X: Tensor, labels: Tensor) -> Tensor:
    """Silhouette score implemented in PyTorch for GPU acceleration

    .. seealso::

        `Wikipedia's entry on Silhouette score <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Parameters
    ----------
    X : Tensor
        Data points of shape (n_samples, n_features)
    labels : Tensor
        Predicted label for each sample, shape (n_samples)

    Returns
    -------
    Tensor
        Silhouette score for each sample, shape (n_samples). Same device and dtype as X
    """
    unique_labels = torch.unique(labels)

    intra_dist = torch.zeros(labels.size(), dtype=X.dtype, device=X.device)
    for label in unique_labels:
        where = labels == label
        X_where = X[where]
        distances = torch.cdist(X_where, X_where)
        intra_dist[where] = distances.sum(dim=1) / (len(distances) - 1)

    inter_dist = torch.full(labels.size(), torch.inf, dtype=X.dtype, device=X.device)

    for label_a, label_b in torch.combinations(unique_labels, 2):
        where_a = labels == label_a
        where_b = labels == label_b

        dist = torch.cdist(X[where_a], X[where_b])
        dist_a = dist.mean(dim=1)
        dist_b = dist.mean(dim=0)

        inter_dist[where_a] = torch.minimum(dist_a, inter_dist[where_a])
        inter_dist[where_b] = torch.minimum(dist_b, inter_dist[where_b])

    sil_samples = (inter_dist - intra_dist) / torch.maximum(intra_dist, inter_dist)

    return sil_samples.nan_to_num()
