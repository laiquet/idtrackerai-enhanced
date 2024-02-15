<h1 align="center">
<img src="https://gitlab.com/polavieja_lab/idtrackerai/-/raw/master/docs/source/_static/logo_neutral.svg" width="400">
</h1><br>

[![image](http://img.shields.io/pypi/v/idtrackerai.svg)](https://pypi.python.org/pypi/idtrackerai/)
![pipeline](https://gitlab.com/polavieja_lab/idtrackerai/badges/master/pipeline.svg)
[![Documentation Status](https://readthedocs.org/projects/idtrackerai/badge/?version=latest)](https://idtracker.ai/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/idtrackerai.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/idtrackerai/)
[![PyPI downloads](https://img.shields.io/pypi/dm/idtrackerai.svg)](https://pypistats.org/packages/idtrackerai)
![Licence](https://img.shields.io/gitlab/license/polavieja_lab/idtrackerai.svg)
[![Nature Paper](https://img.shields.io/badge/DOI-10.1038%2Fs41592--018--0295--5-blue)](
https://doi.org/10.1038/s41592-018-0295-5)

# Find everything you are looking for in [our website](https://idtracker.ai)

Idtracker.ai is a multi-animal tracking software for laboratory conditions. This work has been published in [Nature Methods](https://doi.org/10.1038/s41592-018-0295-5) ([pdf here](https://drive.google.com/file/d/1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n/view?usp=sharing))

## Installation for developers.

On an environment with Python 3.10 or 3.11 and a working installation of Pytorch (Torch and Torchvision) you can install the latest published idtracker.ai version by installing directly form the GitLab repo:

``` bash
pip install git+https://gitlab.com/polavieja_lab/idtrackerai
```

Or install the developing version from the develop branch (currently `v5-dev`):

``` bash
pip install git+https://gitlab.com/polavieja_lab/idtrackerai@v5-dev
```


There exist two extra dependencies options:
 - ``dev`` to install tools for formatting, static analysis, building, publishing, etc.
 - ``docs`` to install needed packages to build documentation (sphinx and some plugins).

## Contributors
* Jordi Torrents (2022-)
* Antonio Ortega (2021-2023)
* Francisco Romero-Ferrero (2015-2022)
* Mattia G. Bergomi (2015-2018)
* Ricardo Ribeiro (2018-2020)
* Francisco J.H. Heras (2015-2022)

***

All present files here are part of idtracker.ai, a project described in:

    Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019. idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.

Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi, Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the Champalimaud Foundation.

idtracker.ai is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. In addition, we require derivatives or applications to acknowledge the authors.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

For more information please send an email (idtrackerai@gmail.com) or use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
