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

from pathlib import Path

from idtrackerai.utils import conf, create_dir


class NetworkParams_crossings:
    # TODO study if identification NetworkParams can be reused here
    def __init__(
        self,
        number_of_classes,
        architecture=None,
        use_adam_optimiser=False,
        restore_folder=None,
        save_folder=None,
        knowledge_transfer_folder=None,
        image_size=None,
        loss="CE",
        print_freq=-1,
        use_gpu=True,
        optimizer="SGD",
        schedule: list[int] = [],
        optim_args=None,
        apply_mask=False,
        dataset=None,
        skip_eval=False,
        epochs=None,
        plot_flag=True,
        saveid="",
        model_name="",
    ):
        if epochs is None:
            epochs = conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN
        self.number_of_classes = number_of_classes
        self.architecture = architecture
        if restore_folder:
            self._restore_folder = Path(restore_folder)
        if save_folder:
            self._save_folder = Path(save_folder)
        if knowledge_transfer_folder:
            self._knowledge_transfer_folder = Path(knowledge_transfer_folder)
        self.use_adam_optimiser = use_adam_optimiser
        self.image_size = image_size
        self.loss = loss
        self.use_gpu = use_gpu
        self.print_freq = print_freq
        self.optimizer = optimizer
        self.schedule = schedule
        self.optim_args = optim_args
        self.apply_mask = apply_mask
        self.dataset = dataset
        self.skip_eval = skip_eval
        self.epochs = epochs
        self.plot_flag = plot_flag
        self.saveid = saveid
        self.model_name = model_name

        if self.optimizer == "SGD":
            self.optim_args["momentum"] = 0.9

    @property
    def restore_folder(self) -> Path:
        return self._restore_folder

    @restore_folder.setter
    def restore_folder(self, path: Path) -> None:
        assert path.is_dir()
        self._restore_folder = path

    @property
    def save_folder(self) -> Path:
        return self._save_folder

    @save_folder.setter
    def save_folder(self, path: Path) -> None:
        create_dir(path)
        self._save_folder = path

    @property
    def knowledge_transfer_folder(self) -> Path:
        return self._knowledge_transfer_folder

    @knowledge_transfer_folder.setter
    def knowledge_transfer_folder(self, path: Path) -> None:
        assert path.is_dir()
        self._knowledge_transfer_folder = path
