# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from idtrackerai.utils import create_dir, json_default


@dataclass
class NetworkParams:
    number_of_classes: int
    schedule: list[int]
    architecture: str
    model_name: str
    dataset: str
    image_size: list[int]
    optim_args: Optional[dict] = field(default_factory=dict)
    scopes_layers_to_optimize: Optional[list[str]] = field(default_factory=list)
    use_adam_optimiser: bool = False
    restore_folder: Path = Path("")
    save_folder: Path = Path("")
    knowledge_transfer_folder: Path | None = None
    loss: str = "CE"
    use_gpu: bool = True
    optimizer: str = "SGD"
    apply_mask: bool = False
    skip_eval: bool = False
    epochs: Optional[int] = None
    return_store_objects: bool = False

    @property
    def load_model_path(self) -> Path:
        return self.restore_folder / (self.model_file_name + ".model.pth")

    @property
    def save_model_path(self) -> Path:
        return self.save_folder / self.model_file_name

    @property
    def model_file_name(self) -> str:
        return f"{self.dataset}_{self.model_name}"

    @property
    def knowledge_transfer_model_file(self) -> Path | None:
        if self.knowledge_transfer_folder is None:
            return None
        return (
            self.knowledge_transfer_folder
            / "supervised_identification_network.model.pth"
        )

    def save(self) -> None:
        path = self.save_folder / "model_params.json"
        logging.info(f"Saving NetworkParams at {path}")
        create_dir(self.save_folder)
        path.write_text(json.dumps(asdict(self), indent=4, default=json_default))
