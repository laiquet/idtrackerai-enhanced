<h1 align="center">
<img src="https://gitlab.com/polavieja_lab/idtrackerai/-/raw/master/docs/source/_static/logo_neutral.svg" width="400">
</h1><br>

[![image](http://img.shields.io/pypi/v/idtrackerai.svg)](https://pypi.python.org/pypi/idtrackerai/)
![pipeline](https://gitlab.com/polavieja_lab/idtrackerai/badges/master/pipeline.svg)
[![Documentation Status](https://readthedocs.org/projects/idtrackerai/badge/?version=latest)](https://idtracker.ai/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/idtrackerai.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/idtrackerai/)
[![PyPI downloads](https://img.shields.io/pypi/dm/idtrackerai.svg)](https://pypistats.org/packages/idtrackerai)
![Licence](https://img.shields.io/gitlab/license/polavieja_lab/idtrackerai.svg)
[![eLife Paper](https://img.shields.io/badge/DOI-10.7554%2FeLife.107602-blue)](https://doi.org/10.7554/eLife.107602)

_idtracker.ai tracks up to 100 unmarked animals from videos recorded in laboratory conditions using artificial intelligence. Free and open source._

This work has been published in [eLife Methods](https://doi.org/10.7554/eLife.107602), please include the following reference if you use this software in your research:

- ```plain
  Jordi Torrents, Tiago Costa, Gonzalo G de Polavieja. New idtracker.ai: rethinking multi-animal tracking as a representation learning problem to increase accuracy and reduce tracking timese. Life14:RP107602 (2025)
  ```
- ```bibtex
  @article{idtrackerai_2025,
            title={New idtracker.ai: rethinking multi-animal tracking as a representation learning problem to increase accuracy and reduce tracking times},
            url={http://dx.doi.org/10.7554/eLife.107602},
            DOI={10.7554/elife.107602},
            publisher={eLife Sciences Publications, Ltd},
            journal={eLife},
            author={Torrents, Jordi and Costa, Tiago and de Polavieja, Gonzalo G},
            year={2025}
          }
  ```

Visit [our website](https://idtracker.ai) to find more information about the software, installation instructions, and user guides.

## Installation for developers

On an environment with Python>=3.10 and a working installation of Pytorch (Torch and Torchvision) you can install the latest published idtracker.ai version by installing directly form the GitLab repo:

```bash
pip install git+https://gitlab.com/polavieja_lab/idtrackerai
```

Or install the developing version from the develop branch:

```bash
pip install git+https://gitlab.com/polavieja_lab/idtrackerai@develop
```

There exist two extra dependencies options:

- `dev` to install tools for formatting, static analysis, building, publishing, etc.
- `docs` to install needed packages to build documentation (sphinx and some plugins).

## Contributors

- Jordi Torrents (2022-)
- Tiago Costa (2024)
- Antonio Ortega (2021-2023)
- Francisco Romero-Ferrero (2015-2022)
- Mattia G. Bergomi (2015-2018)
- Ricardo Ribeiro (2018-2020)
- Francisco J.H. Heras (2015-2022)

---

For more information please send an email (info@idtracker.ai) or use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.
