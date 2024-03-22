Output structure
================

Idtracker.ai will generate a ``session_[SESSION_NAME]`` folder in the same directory where the input videos are (or in the ``--output_dir`` path if specified, see :ref:`output`). The session folder may be have the following structure:

.. admonition:: Note
    :class: sidebar note

    The content of the session folder may change depending on the necessities of each session. Also, ``--data_policy`` can remove some of the data after finishing a successful tracking (see :ref:`output`).

.. code-block::
    :caption: idtracker.ai session's output structure
    :emphasize-lines: 18-30

    session_[SESSION_NAME]
    ├─ accumulation_*
    │  ├─ identification_network.model.pth
    │  ├─ list_of_fragments.json
    │  └─ model_params.json
    ├─ crossings_detector
    │  └─ crossing_detector.model.pth
    ├─ bounding_box_images
    │  └─ bbox_images_*.hdf5
    ├─ identification_images
    │  └─ id_images_*.hdf5
    ├─ preprocessing
    │  ├─ background.png
    │  ├─ list_of_blobs.pickle
    │  ├─ list_of_fragments.json
    │  ├─ list_of_global_fragments.json
    │  └─ ROI_mask.png
    ├─ trajectories
    │  ├─ with_gaps_csv
    │  │  ├─ areas.csv
    │  │  ├─ id_probabilities.csv
    │  │  ├─ trajectories.csv
    │  │  └─ attributes.json
    │  ├─ without_gaps_csv
    │  │  ├─ areas.csv
    │  │  ├─ id_probabilities.csv
    │  │  ├─ trajectories.csv
    │  │  └─ attributes.json
    │  ├─ with_gaps.npy
    │  └─ without_gaps.npy
    ├─ session.json
    └─ idtrackerai.log

The trajectories folder has been highlighted above, it contain the most valuable data for the end user, the position of every animal in every video frame. See how to read them in :ref:`trajectory files`.

In the session folder there's a copy of the session log ``idtrackerai.log`` made at the end of the process (successful or not). This file contains information of the entire tracking process and is essential to debug and understand idtracker.ai (see :ref:`tracking log`).

The majority of the generated data is a byproduct of the tracking process and it is not meant to be read or used by the end user. Still, an intuition of the data content can be read as:

- ``accumulation_*`` contains the identification network model and parameters. It can be used to match identities with other sessions with :ref:`idmatcher.ai`.
- ``crossings_detector`` contains the individual/crossing classification network model and parameters.
- ``identification_images`` contains the images used for identification. This is, an image for every animal and every frame on the video.
- ``preprocessing`` contains the blobs, fragments and global fragments objects (in Python pickle format). Advanced users can dive into these objects to gather some extra information about the tracking. Also, the ROI and the computed background are saved here.
- ``segmentation_data`` contains the temporal image used to generate the final identification images.
- ``session.json`` contains basic properties of the video and the session in human readable *.json* format.


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
- ``length_unit``: ratio between the pixel distance and the real distance stated by the user of all pairs of points defined using the :ref:`length calibration` tool.

.. warning::
    ``body_length`` is not a reliable measurement of the real size of the animal. Its value depends on the segmentation parameters and the video conditions.

Types of trajectory files
=========================

When crossings occur, the identification network cannot be applied and the involved individuals cannot be located properly. In these situations, the trajectories have a *gap* full of :abbr:`NaN (Not a number)` values, i.e. the individual couldn't be located. These trajectories are saved in ``with_gaps.npy``.

To close the gaps, an interpolation algorithm takes place and generates an improved ``without_gaps.npy`` file where most of the gaps have been closed. Some gaps are difficult to close and there's no guarantee for ``without_gaps.npy`` not to contain any *NaN* gap.

When tracking without identities, the trajectories will be saved only in ``with_gaps.npy``. Since there are random identity assignments, the interpolation algorithm for closing gaps cannot be applied.

Finally, if the :ref:`validator` is used after the tracking, the ``validated.npy`` file will contain the trajectories manually corrected by the user.
