:sd_hide_title:

************
Installation
************

.. note::
    If you encounter problems during installation, send an email to idtrackerai@gmail.com. We will try our best to assist you.

Requirements
============

Idtracker.ai is a Python package (uploaded to `PyPI <https://pypi.org/project/idtrackerai/>`_) tested on Linux (Mint and Ubuntu) and Windows. Currently, we don't give support for macOS but, as all the components of idtracker.ai work on macOS, idtracker.ai should too (at your own risk).

Idtracker.ai uses neural networks to track and identify animals, for which it depends on Pytorch. That's why **to run idtracker.ai's tracking algorithms, a dedicated Nvidia GPU is required**. If your machine does **not** have a dedicated NVIDIA GPU, you still can use some of the tools idtracker.ai offers; see :ref:`install without a nvidia gpu`.

.. admonition:: Heavy videos
    :class: sidebar warning

    Tracking and working with heavy videos (4K resolution, >10min duration, >20 animals) may have higher requirements, especially in RAM.

Besides the neural networks, idtracker.ai is a resource consuming software so it is recommended to run on a moderately equipped computer. The following is the recommended minimum configuration:


.. grid:: 1 2 2 2

    .. grid-item::

            - 12GB RAM memory

    .. grid-item::

            - 100GB free space

    .. grid-item::

            - Intel i5 or equivalent

    .. grid-item::

            - 2GB GPU memory

Check Nvidia drivers
====================

idtracker.ai depends on PyTorch which works with :abbr:`Cuda (Nvidia's language that allows other software to use the GPU)` >= 11.7 . Assuming you computer is using a Nvidia GPU, you need Cuda >= 11.7. Check your current NVIDIA drivers installation by opening a terminal (Anaconda prompt on Windows) and typing:

.. code-block:: bash

    nvidia-smi

to get an output similar to this:

.. code-block::
    :caption: ``nvidia-smi`` output
    :name: nvidia-smi output

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.78.01    Driver Version: 525.78.01    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
    | N/A   60C    P0    N/A /  35W |      5MiB /  4096MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      2186      G   /usr/lib/xorg/Xorg                  4MiB |
    +-----------------------------------------------------------------------------+


Check your Cuda version in the part "*CUDA Version:*", if it is equal or higher than 11.7, you can go to the next installation step, :ref:`check conda environments`.

If your Cuda version is lower than 11.7 (or you don't get the :ref:`nvidia-smi output` at all) you need to update (or install) the Nvidia drivers in your machine.

.. tip::
    As a rule of thumb, avoid manually installing critical drivers like Nvidia's ones. Let your operating system update them automatically.


.. tab-set::

    .. tab-item:: For Ubuntu users

        Give Ubuntu a chance to install drivers by its own by running a general update with:

        .. code-block:: bash

            sudo apt update
            sudo apt upgrade

        and reboot if asked.

        If the :ref:`nvidia-smi output` stays the same, open Ubuntu's application *Software & Updates*  (if you don't find it on your applications, you can launch it running the command ``software-properties-gtk``)

        .. image:: ../_static/software&updates_dark.png
            :class: only-dark

        .. figure:: ../_static/software&updates_light.png
            :class: only-light

            Ubuntu's *Software & Updates* application

        In the tab *Additional Drivers*, select the NVIDIA driver **(proprietary, tested)** and click *Apply Changes*. Wait the installation to finish and reboot when asked.

    .. tab-item:: For Windows users

        Give Windows a chance to install drivers by its own by running a general update with *Windows Update*, you can run it with the command

        .. code-block:: bash

            control update

        This command will launch a graphical application, check for updates there and install. Reboot when asked.

        If the :ref:`nvidia-smi output` stays the same, open Nvidia's application *GeForce Experience* (or install it from `their website <https://www.nvidia.com/en-us/geforce/geforce-experience/>`_ :fa:`fa-solid fa-arrow-up-right-from-square`).

        .. figure:: ../_static/GeForceExperience.png
            :class: dark-light

            Nvidia's *GeForce Experience* application

        In the tab *DRIVERS*, click *CHECK FOR UPDATES*. Update your drivers and reboot when asked. If everything fails, you can still try to manually install drivers from `Nvidia website <https://www.nvidia.com/Download/index.aspx>`_ :fa:`fa-solid fa-arrow-up-right-from-square`.

Check Conda environments
========================

While it is not required, we recommend installing idtracker.ai inside a Conda environment. You can check if you have a Conda installation by running

.. code-block:: bash

    conda

If you get ``conda: command not found``, you do **not** have Conda installed. Its installation is easy, follow the `Conda installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_ :fa:`fa-solid fa-arrow-up-right-from-square`.

.. tip::
    When deciding whether to install Anaconda or Miniconda, read `their section <https://conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_ :fa:`fa-solid fa-arrow-up-right-from-square` about their differences. If you are not sure, we recommend Miniconda.


Install idtracker.ai
====================

Assuming you have NVIDIA Cuda >= 11.7 and Anaconda (or Miniconda) on your system, idtracker.ai can be now installed by following the commands below (to be run in a Linux terminal or in an Anaconda Prompt in Windows):

1. Create a Conda environment called *idtrackerai* with Python 3.10 (modify the name of the environment if desired):

   .. code-block::

    conda create -n idtrackerai python=3.10

2. Activate the environment:

   .. code-block::

    conda activate idtrackerai

3. Install idtracker.ai from `PyPI <https://pypi.org/project/idtrackerai/>`_:

   .. code-block::

    python -m pip install idtrackerai

4. Go to `PyTorch site <https://pytorch.org/get-started/locally/#start-locally>`_ to get the command to install `Pytorch` and `Torchvision` with the parameters *Pytorch Build: Stable*, your operating system, *Package: Conda*, *Language: Python* and your Cuda version (for Cuda > 11.8, select *CUDA 11.8*). The command will appear as:

   .. code-block:: bash

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   This command depends on your computer specifications, don't copy-paste it, visit `PyTorch site <https://pytorch.org/get-started/locally/#start-locally>`_.

If you are unfamiliar with Conda environments, keep in mind that idtracker.ai has been installed **inside a Conda environment**. As long as the environment is inactive, your computer will have no idea about any idtrackerai installation. To run any idtracker.ai command in the future you will have to activate first the environment with:

.. code-block:: 

    conda activate idtrackerai

Install without a NVIDIA GPU
============================

Use idtrackerai without Pytorch
-------------------------------

The :ref:`segmentation app`, the :ref:`validator` and the :ref:`video generators` do **not** require Pytorch and, hence, they do not need a dedicated Nvidia GPU. You can use these tools by installing **only** the steps 1 to 3 of :ref:`install idtracker.ai`.

This kind of installation can be useful to control a full installation located in a remote computer. You can prepare your input parameters on your local machine, run the tracking on remote and validate and process the output in your local machine again.

Install Pytorch with AMD GPU
----------------------------

While we don't give support for it, you still can install Pytorch (and therefore idtracker.ai) with an AMD GPU with their API *ROCm* (Ubuntu, Linux, Red Hat, and CentOS only). Follow the steps 1 to 3 of :ref:`install idtracker.ai` and then install Pytorch by selecting *Compute Platform: ROCm* in `their site <https://pytorch.org/get-started/locally/#start-locally>`_.

Install Pytorch with MacOS
--------------------------

While we don't give support for it, you still can install Pytorch (and therefore idtracker.ai) in a Mac computer (MacOS >= 12.3). Follow the steps 1 to 3 of :ref:`install idtracker.ai` and install Pytorch by selecting *Your OS: Mac* in `their site <https://pytorch.org/get-started/locally/#start-locally>`_.

Install Pytorch for CPU
-----------------------

You still can install Pytorch (and therefore idtracker.ai) to run in your CPU (Linux and Windows only). Follow the steps 1 to 3 of :ref:`install idtracker.ai` and install Pytorch by selecting *Compute Platform: CPU* in `their site <https://pytorch.org/get-started/locally/#start-locally>`_.

This installation can be useful if you want to track a single animal, or to :ref:`track without identities`. In these cases the identification algorithms are not used and you won't notice the lack of a proper GPU. However, if you want to track multiple animals with identification, the neural networks algorithms will run desperately slow in your CPU making this installation unusable.

Test the installation
=====================

Open a terminal (Anaconda Prompt in Windows) and activate the Conda environment where your idtracker.ai installation is:

.. code-block:: bash

    conda activate idtrackerai

Test your idtracker.ai installation by running:

.. code-block:: bash

    idtrackerai_test

.. admonition:: Manual download
    :class: sidebar note

    :download:`test_B.avi <https://drive.google.com/uc?export=download&id=1T5acLY9eBLpD4ZR3O4TWpC3k0pcZ9POU>`

This command will copy a 18 seconds test video called ``test_B.avi`` into you current working directory and idtracker.ai will track it generating the respective ``session_test`` output folder.

.. admonition:: Not recognized command
    :class: warning


    If just installed idtracker.ai and this test gets a very short error like ``No such file or directory``, try reactivating the Conda environment:

    .. code-block:: bash

        conda deactivate
        conda activate idtrackerai


.. admonition:: Could not load library libcudnn_cnn_infer.so.8
    :class: warning

    If the test starts but after some seconds you get something like ``Could not load library libcudnn_cnn_infer.so.8``. Install the Cuda toolkit dependency:

    .. code-block:: bash

        conda install cudatoolkit=11.8 -c conda-forge

With GPU support, the test takes from 2 to 6 minutes. Without it (:ref:`install pytorch for cpu`), it takes up to 20-60 minutes. At the end of the test, the console should display the following line:

.. parsed-literal::

    INFO     Test passed successfully in ? min with version |version|

meaning a successful installation! :fa:`fa-solid fa-face-laugh`

.. seealso::
    A 4K resolution and 1 minute long video with 100 zebrafish are also available in `Google Drive <https://drive.google.com/open?id=1Tl64CHrQoc05PDElHvYGzjqtybQc4g37>`_ for users to test idtracker.ai's capabilities on a more demanding video.

Update idtracker.ai
===================

From 5.x
--------

To update idtracker.ai from version 5.x to current version |version|, run (inside the environment):

.. code-block:: bash 

    python -m pip install --upgrade idtrackerai

From 4.x or below
-----------------

To update idtracker.ai from version 4.x (or below) to current version |version|, you will have to :ref:`uninstall` the old conda environment and install the new version from scratch as version 4.x and 5.x use different Python versions.

Uninstall
=========

To remove everything inside a Conda environment and the environment itself, from outside the environment run:

.. code-block:: bash

    conda remove -n idtrackerai --all