Trajectory files
================

The most important files for the end user are the trajectory files, located in the folder `trajectories`. Once the tracking process finishes successfully, trajectory files can be loaded in Python with

.. code-block:: python

    import numpy as np

    trajectories_dict = np.load(
        "./session_example/trajectories/trajectories_wo_gaps.npy", allow_pickle=True
    ).item()

.. tip::
    *.npy* files can only be loaded with Numpy (Python). If you want idtracker to automatically convert theses files into *.csv* and *.json* files, set ``CONVERT_TRAJECTORIES_TO_CSV_AND_JSON`` to ``true`` before running idtracker.ai (see :ref:`advanced parameters<output>`).

    If you missed it and the tracking is done, you still can convert those files by running

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

When crossings occur, the identification network cannot be applied and the involved individuals cannot be located properly. In these situations, the trajectories have a *gap* full of :abbr:`NaN (Not a number)` values, i.e. the individual couldn't be located. These trajectories are saved in ``trajectories.npy``.

To close the gaps, an interpolation algorithm takes place and generates an improved ``trajectories_wo_gaps.npy`` file where most of the gaps have been closed. Some gaps are difficult to close and there's no guarantee for ``trajectories_wo_gaps.npy`` not to contain any *NaN* gap.

Finally, if the :ref:`validator` is used after the tracking, the ``trajectories_validated.npy`` file will contain the trajectories manually corrected by the user.
