import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from idtrackerai.utils import create_dir, json_default, resolve_path


@dataclass(slots=True)
class NetworkParams:
    n_classes: int
    schedule: list[int]
    model_name: str
    image_size: list[int]
    architecture: str = "CNN"
    optim_args: dict = field(default_factory=dict)
    epochs: int = 0
    optimizer: Literal["Adam", "SGD"] = "SGD"
    loss: str = "CE"
    save_folder: Path = field(default_factory=Path)
    knowledge_transfer_folder: Path | None = None
    restore_folder: Path = field(default_factory=Path)

    @classmethod
    def from_file(cls, path: str | Path):
        path = resolve_path(path)
        logging.info("Loading Network Params from %s", path)
        if path.is_dir():
            path /= "model_params.json"
        with path.open() as file:
            params = json.load(file)
        network_params = cls(**params)
        network_params.save_folder = Path(network_params.save_folder)
        if network_params.knowledge_transfer_folder is not None:
            network_params.knowledge_transfer_folder = Path(
                network_params.knowledge_transfer_folder
            )
        network_params.restore_folder = Path(network_params.restore_folder)
        return network_params

    @property
    def load_model_path(self) -> Path:
        # v5.0.0 compatibility
        for deprecated_name in (
            self.model_name + "_.model.pth",
            "supervised_" + self.model_name + ".model.pth",
            "supervised_" + self.model_name + "_.model.pth",
        ):
            path = self.restore_folder / deprecated_name
            if path.is_file():
                return path

        return self.restore_folder / (self.model_name + ".model.pth")

    @property
    def model_path(self) -> Path:
        return (self.save_folder / self.model_name).with_suffix(".model.pth")

    @property
    def penultimate_model_path(self) -> Path:
        return (self.save_folder / (self.model_name + "_penultimate")).with_suffix(
            ".model.pth"
        )

    @property
    def knowledge_transfer_model_file(self) -> Path | None:
        if self.knowledge_transfer_folder is None:
            return None
        return self.knowledge_transfer_folder / "identification_network.model.pth"

    def save(self) -> None:
        path = self.save_folder / "model_params.json"
        logging.info(f"Saving NetworkParams at {path}")
        create_dir(self.save_folder)
        json.dump(asdict(self), path.open("w"), indent=4, default=json_default)
