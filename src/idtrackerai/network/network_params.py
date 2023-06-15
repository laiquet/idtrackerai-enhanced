import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

from idtrackerai.utils import create_dir, json_default


@dataclass(slots=True)
class NetworkParams:
    number_of_classes: int
    schedule: list[int]
    architecture: str
    model_name: str
    image_size: list[int]
    optim_args: dict = field(default_factory=dict)
    epochs: int = 0
    optimizer: Literal["Adam", "SGD"] = "SGD"
    loss: str = "CE"
    save_folder: Path = Path("")
    scopes_layers_to_optimize: Optional[list[str]] = field(default_factory=list)
    knowledge_transfer_folder: Path | None = None
    use_adam_optimiser: bool = False
    return_store_objects: bool = False
    restore_folder: Path = Path()

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
