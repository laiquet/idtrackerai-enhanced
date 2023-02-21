************
Installation
************
 
Idtracker.ai is a Python package (available at `PyPI <https://pypi.org/project/idtrackerai/>`_) that runs with Python 3.10 and uses the last versions of PyQT for its apps, OpenCV and NumPy for image processing and PyTorch for neural networks training.

Requirements
============

Idtracker.ai is tested on Linux and Windows. The tools that idtracker.ai offers (segmentation app, validator, video generators...) do not have any specific hardware requirements. However, to run the main program (tracking with identification) in a decent time, a dedicated Nvidia GPU is required.

To have a good experience, it is recommended to have, at minimum:

- 16GB RAM memory
- 100GB free hard drive space,
- GPU: Nvidia TITAN X / GeForce GTX 1060

Pre-installation checks
=======================

Check NVIDIA drivers 
--------------------


idtracker.ai uses the last version of PyTorch so Cuda >= 11.6 is required. Before installing check which NVIDIA driver you have installed and its compatibility with the corresponding CUDA toolkit version (see `cuda compatiblity <https://docs.nvidia.com/deploy/cuda-compatibility/>`).


To check whether the NVIDIA drivers are correctly installed in your computer, 
open a terminal (Anaconda prompt on Windows) and type:

.. code-block:: bash

    nvidia-smi

You should get an output similar to this one

.. code-block::
    :caption: nvidia-smi output

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.78.01    Driver Version: 525.78.01    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:08:00.0  On |                  Off |
    |  0%   34C    P8    12W / 450W |    543MiB / 24564MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     53796      G   /usr/lib/xorg/Xorg                204MiB |
    |    0   N/A  N/A     53898      G   /usr/bin/gnome-shell               36MiB |
    |    0   N/A  N/A     54794      G   ...RendererForSitePerProcess      152MiB |
    |    0   N/A  N/A     56154      G   ...1/usr/lib/firefox/firefox      147MiB |
    +-----------------------------------------------------------------------------+


Check that in the part where it says "Driver Version" you have value higher 
than 440.33 (compatible with CUDA 10.2) or 450.80.02 (compatible with CUDA 11.3).


If you fail to get this output or your version is smaller than 440.33, 
then you will need to instal or update your nvidia drivers.

> NOTE: `this link <https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/>`__
> has nice instructions to get the latest NVIDIA drivers either using your Update Manager or the terminal.

1. Clean the system of other Nvidia drivers

.. code-block:: bash

    sudo apt-get purge nvidia*

2. Check which is the latest driver version system in `this link <https://www.nvidia.com/object/unix.html>`__.

3. Update and upgrade your system:

.. code-block:: bash

    sudo apt update
    sudo apt upgrade

1. Check which is the latest available version of the NVIDIA drivers for your system:

.. code-block:: bash

    apt search nvidia-driver

5. Install the NVIDIA GPU driver. In the following command, substitute the XXX by the number of the driver you want to install (e.g. `nvidia-driver-495`).

.. code-block:: bash

    sudo apt-get install nvidia-driver-XXX

6. Reboot the system.

.. code-block:: bash

    sudo reboot

7. Check the installation.

.. code-block:: bash

    nvidia-smi

For Windows users
*****************

To check which NVIDIA drivers you have installed in your computer following these steps
(adapted from `this page <https://www.drivereasy.com/knowledge/how-to-check-nvidia-driver-version-easily/>`_):

1. Right click any empty area on your desktop screen, and select NVIDIA Control Panel.

2. Click System Information (on the bottom left corner) to open the driver information.

3. Check the Driver version in the Details section.

You can download the latest driver available for your GPU from `the NVIDIA webpage <https://www.nvidia.com/Download/index.aspx>`_.

After downloading the *.exe* file, execute it and follow the instructions.
After the installation you will be asked to reboot the computer, please do so for the installation to be complete.

> NOTE: For Windows you will need an NVIDIA driver >=441.22 for CUDA 10.2 and >=456.38 for CUDA 11.3.

Preparing a Conda environment (for Linux and Windows)
-----------------------------------------------------

It is good practice to install python packages in virtual environments. In particular,
we recommend using Conda virtual environments. Find here the `Conda installation
instructions for Linux and Windows <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

When deciding whether to install Anaconda or Miniconda, you can find some information about the differences
`here <https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda>`__. For simplicity, we recommend
installing Miniconda.

From now on, every time we refer to the *terminal*, Linux users are meant to use the command line and Windows user
are meant to use the Anaconda Powershell Prompt that it is installed when installing Miniconda or Anaconda.

To check whether the Conda package manager is installed, you can open a terminal and type

.. code-block:: bash

    conda

if you get the following output

.. code-block:: bash

    conda: command not found

Miniconda is not installed in your system. Follow the instructions in the link above to install it.


Install
=======

Assuming that you have the latest version of the NVIDIA drivers installed, and Anaconda (or Miniconda) installed, idtracker.ai can be installed by following the commands below (to be run in a linux terminal or in an Anaconda Powershell Prompt in Windows):

.. code-block:: bash

    conda create -n idtrackerai python=3.10
    conda activate idtrackerai
    pip install idtrackerai

To use the main tracking program of idtracker.ai (tracking with identities), go to `PyTorch site <https://pytorch.org/get-started/locally/#start-locally>`_ to install `pytorch` and `torchvision`, the command will appear as

.. code-block:: bash

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

.. warning:: 
    This command depends on you OS and CUDA version. Don't copy-paste it, visit `PyTorch site <https://pytorch.org/get-started/locally/#start-locally>`_

For running the segmentation app, the validator app, to generate videos or to read idtracker.ai's output, you do not need to install PyTorch. You only need PyTorch to run the tracking.


Test the installation
=====================

Open a terminal (Anaconda Prompt in Windows) and activate the Conda environment where you installed idtracker.ai.

.. code-block:: bash

    conda activate [NAME_OF_THE_ENVIRONMENT]

If you don't remember the name of the environment, you can type :code:`conda env list` to list all the environments in your computer.

Once done, you can test your installation by running:

.. code-block:: bash

    idtrackerai_test

This command will run idtracker.ai in an internal 18 seconds video.

If you want to access this internal video and the test results, you can add run

.. code-block:: bash

    idtrackerai_test -o /path/to/the/save_folder

to run the test and copy move the input video and the output generated data to the specified location.


In an installation with GPU support the test takes from 3 to 6 minutes. Running with no-GPU support it can take up to 20-60 minutes. At the end of the test, the console should display something like  

.. code-block:: 

    INFO     Estimated accuracy: 99.6377%
    INFO     Data policy: all
    INFO     Success


A 4K resolution, 1 minute and 100 fish video is available `here <https://drive.google.com/open?id=1Tl64CHrQoc05PDElHvYGzjqtybQc4g37>`_ for users to test idtracker.ai's capabilities.

Uninstall
=========

To remove everything inside a Conda environment and the environment itself, from outside the environment run

.. code-block:: bash

    conda remove -n name-of-the-environment --all
