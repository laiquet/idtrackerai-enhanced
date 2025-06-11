****************************
Installation Troubleshooting
****************************

Issues with Qt
==============

If you have any issue with the default Qt binding (PyQt6) you can switch to another Qt binding for Python:

1. Uninstall the default PyQt6 if it was installed:

   .. code-block:: bash

       python -m pip uninstall PyQt6

2. Install **one** of the other Qt bindings for Python:

   .. code-block:: bash
       :caption: Alternative Qt bindings for Python

       # choose one of the following options
       python -m pip install PyQt5
       python -m pip install PySide6
       python -m pip install PySide2

3. Install idtracker.ai without the PyQt6 dependency in case it was not installed yet:

   .. code-block:: bash

       # install idtrackerai without any dependency
       python -m pip install idtrackerai --no-deps
       # install the rest of the dependencies except PyQt6
       python -m pip install numpy rich h5py scipy opencv-python-headless qtpy superqt toml matplotlib packaging psutil scikit-learn requests deprecated

.. admonition:: Note

    If no Qt binding works on your computer, you can still track directly from a *toml* file without running the GUI with the ``--track`` flag (see :ref:`terminal usage`) and use the non-GUI :ref:`tools`.

Not recognized command
----------------------

If you have just installed idtracker.ai and encounter a short error like ``No such file or directory`` or ``Not recognized command`` when running ``idtrackerai_test``, try reactivating the Conda environment:

.. code-block:: bash

    conda deactivate
    conda activate idtrackerai


Could not load library libcudnn_cnn_infer.so.8
----------------------------------------------

If ``idtrackerai_test`` starts but after a few seconds you see an error like ``Could not load library libcudnn_cnn_infer.so.8``, it means you are missing the Cuda Toolkit dependency. Install it with:

.. code-block:: bash

    conda install cudatoolkit=11.8 -c conda-forge

qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
----------------------------------------------------------

Refer to :external:`this thread <https://forum.qt.io/topic/93247/qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it-was-found?sort=most_votes>` for more information. Alternatively, on Ubuntu, you can solve the problem by running ``sudo apt install libxcb-cursor0``.


module 'torch' has no attribute '_six'
--------------------------------------

In MacOS, running

.. code-block:: bash

    python -m pip install --upgrade torch

seems to fix the issue.

ImportError: cannot import name ``COMMON_SAFE_ASCII_CHARACTERS`` from ``charset_normalizer.constant``
-----------------------------------------------------------------------------------------------------

From :external:`Stack Overflow <https://stackoverflow.com/questions/74535380>`:

.. code-block:: bash

    python -m pip install chardet


No graphic device was found available
-------------------------------------

If your computer has an NVIDIA or AMD GPU, or uses MacOS >= 12.3 with an M1, M2, or AMD GPU, and idtrackerai cannot find the specific device available, it means your PyTorch installation is malfunctioning.

To fix this, you need to :ref:`uninstall` the entire Conda environment and try again. Carefully read the :external:`PyTorch instructions <https://pytorch.org/get-started/locally/>` based on your machine. Getting PyTorch to use GPU acceleration can sometimes be tricky.


.. admonition:: Any other error
    :class: note

    Send us your error to info@idtracker.ai and we will assist you.
