from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class NetworkParams:
    # TODO remove completely this class
    n_classes: int
    model_name: str
    image_size: list[int]
    save_folder: Path = field(default_factory=Path)

    @property
    def model_path(self) -> Path:
        return (self.save_folder / self.model_name).with_suffix(".model.pth")

    @property
    def penultimate_model_path(self) -> Path:
        return (self.save_folder / (self.model_name + "_penultimate")).with_suffix(
            ".model.pth"
        )
