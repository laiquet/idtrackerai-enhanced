****************************
Installation Troubleshooting
****************************


Cannot install PyQt5 dependency
-------------------------------

Idtrackerai's GUIs work (by default) with PyQt5. But if its installation fails, giving some of the next errors:

- **sipbuild.exceptions.UserException**
- **error: metadata-generation-failed**
- **sipbuild.pyproject.PyProjectOptionException**
- Frozen process while ``Preparing metadata (pyproject.toml)...`` (specially on MacOS)

you can choose to install any of the remaining Qt bindings for Python (you only need **one** of them to succeed):

.. admonition:: Note
    :class: sidebar note

    If any Qt installation works on your computer, you still can use idtrackerai non-GUI parts by following the next commands without any Qt installation.

.. code-block:: bash
    :caption: Qt binding options in Python

    python -m pip install PyQt5 # default
    python -m pip install PyQt6
    python -m pip install PySide6
    python -m pip install PySide2

And, once any of them succeed, you will have to install idtrackerai without the PyQt5 dependency:

.. code-block:: bash

    # install idtrackerai without any dependency
    python -m pip install idtrackerai --no-deps

    # install all remaining dependencies except PyQt5
    python -m pip install numpy rich h5py scipy opencv-python-headless qtpy superqt toml matplotlib


Not recognized command
----------------------

If you just installed idtracker.ai and ``idtrackerai_test`` gets a very short error like ``No such file or directory`` or ``Not recognized command``, try reactivating the Conda environment:

.. code-block:: bash

    conda deactivate
    conda activate idtrackerai


Could not load library libcudnn_cnn_infer.so.8
----------------------------------------------

If ``idtrackerai_test`` starts but after some seconds you get something like ``Could not load library libcudnn_cnn_infer.so.8``, you are missing the Cuda Toolkit dependency, install it with:

.. code-block:: bash

    conda install cudatoolkit=11.8 -c conda-forge

qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
----------------------------------------------------------

Read :external:`this thread <https://forum.qt.io/topic/93247/qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it-was-found?sort=most_votes>`. Alternatively, in Ubuntu ``sudo apt install libxcb-cursor0`` solves the problem.


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

If your computer has a NVIDIA or AMD GPU or uses MacOS >= 12.3 with M1, M2 or AMD GPU and idtrackerai cannot find the specific device available, your PyTorch installation is malfunctioning.

To fix that, you have to :ref:`uninstall` the entire Conda environment and try again. Read carefully the :external:`PyTorch indications <https://pytorch.org/get-started/locally/>` depending on your machine. Getting PyTorch to use GPU acceleration can be tricky sometimes.


.. admonition:: Any other error
    :class: note

    Send us your error to idtrackerai@gmail.com and we will assist you.
