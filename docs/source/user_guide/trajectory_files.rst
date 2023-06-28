Trajectory files
================

The most useful files for the end user are the trajectory files, located in the folder `trajectories`. The main ones are the binary *.npy* formatted files and, once the tracking process finishes successfully, they can be loaded in Python with:

.. code-block:: python

    import numpy as np

    trajectories_dict = np.load(
        "./session_example/trajectories/without_gaps.npy", allow_pickle=True
    ).item()

Since *.npy* files can only be loaded with Numpy (Python). Idtrackerai automatically generates a copy of these files in human readable *.csv* and *.json* formats.

.. tip::
    If you have an old session with its trajectory files not translated to *.csv*, you still can convert these files by running

    .. code-block:: bash

        idtrackerai_csv path/to/session_[SESSION_NAME]

The *.npy* files contain a Python dictionary with the following keys:

- ``trajectories``: Numpy array with shape (`N_frames`, `N_animals`, 2) with the `xy` coordinate for each identity and frame in the video.
- ``version``: idtracker.ai version which created the current file.
- ``video_paths``: input video paths.
- ``frames_per_second``: input video frame rate.
- ``body_length``: mean body length computed as the mean value of the diagonal of all individual blob's bounding boxes.
- ``stats``: dictionary containing four different measurements of the session's identification accuracy.
- ``areas``: dictionary containing the mean, median and standard deviation of the blobs area for each individual.
- ``setup_points``: dictionary of the user defined setup points (from validator).
- ``identities_labels``: list of user defined identity labels (from validator).
- ``identities_groups``: list of user defined identity groups (from validator).
- ``id_probabilities``: Numpy array with shape (`N_frames`, `N_animals`) with the identity assignment probability for each individual and frame of the video.

.. warning::
    ``body_length`` is not a reliable measurement of the real size of the animal. Its value depends on the segmentation parameters and the video conditions.

Types of trajectory files
=========================

When crossings occur, the identification network cannot be applied and the involved individuals cannot be located properly. In these situations, the trajectories have a *gap* full of :abbr:`NaN (Not a number)` values, i.e. the individual couldn't be located. These trajectories are saved in ``with_gaps.npy``.

To close the gaps, an interpolation algorithm takes place and generates an improved ``without_gaps.npy`` file where most of the gaps have been closed. Some gaps are difficult to close and there's no guarantee for ``without_gaps.npy`` not to contain any *NaN* gap.

When tracking without identities, the trajectories will be saved only in ``with_gaps.npy``. Since there are random identity assignments, the interpolation algorithm for closing gaps cannot be applied.

Finally, if the :ref:`validator` is used after the tracking, the ``validated.npy`` file will contain the trajectories manually corrected by the user.
