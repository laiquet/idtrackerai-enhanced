================
Output structure
================

Idtracker.ai will generate a ``session_[SESSION_NAME]`` folder in the same directory where the input videos are. If specified, it will be in the ``--output_dir`` path (see :ref:`output`). The session folder may have the following structure:

.. admonition:: Note
    :class: sidebar note

    The content of the session folder may vary depending on the requirements of each session. Also, ``--data_policy`` can remove some of the data after finishing a successful tracking (see :ref:`output`).

.. code-block::
    :caption: idtracker.ai session's output structure
    :emphasize-lines: 18-24

    session_[SESSION_NAME]
    ├─ accumulation
    │  ├─ identifier_[cnn, contrastive].model.pt
    │  └─ model_params.json
    ├─ crossings_detector
    │  └─ crossing_detector.model.pth
    ├─ bounding_box_images
    │  └─ bbox_images_*.h5
    ├─ identification_images
    │  └─ id_images_*.h5
    ├─ preprocessing
    │  ├─ background.png
    │  ├─ list_of_blobs.pickle
    │  ├─ list_of_fragments.json
    │  ├─ list_of_global_fragments.json
    │  └─ ROI_mask.png
    ├─ trajectories
    │  ├─ trajectories_csv
    │  │  ├─ areas.csv
    │  │  ├─ id_probabilities.csv
    │  │  ├─ trajectories.csv
    │  │  └─ attributes.json
    │  ├─ trajectories.h5
    │  └─ trajectories.npy
    ├─ session.json
    └─ idtrackerai.log

The trajectories folder has been highlighted above. It contains the most valuable data for the end user: the position of every animal in every video frame. See how to read them in :ref:`trajectory files`.

A copy of the session log ``idtrackerai.log`` is saved in the session folder at the end of the process (whether successful or not). This file contains information of the entire tracking process and is essential to debug and understand idtracker.ai (see :ref:`tracking log`).

Most of the generated data is a byproduct of the tracking process and is not intended for end-user consumption. Still, an intuition of the data content can be read as:

- ``accumulation`` contains the identification network model and parameters. It can be used to match identities with other sessions with :ref:`idmatcher.ai`.
- ``crossings_detector`` contains the individual/crossing classification network model and parameters.
- ``identification_images`` contains the images used for identification. This is, an image for every animal and every frame on the video.
- ``preprocessing`` contains the blobs, fragments and global fragments objects (in Python's Pickle format). Advanced users can dive into these objects to gather some extra information about the tracking. Also, the ROI and the computed background are saved here.
- ``segmentation_data`` contains the temporal image used to generate the final identification images.
- ``session.json`` contains basic properties of the video and the session in human readable *.json* format.

================
Trajectory files
================

The most useful files for the end user are the trajectory files, located in the folder ``/trajectories``. These can be saved in multiple formats following the parameters indicated in :ref:`output`.

These files contain a dictionary-like structure with the following keys:

- ``trajectories``: Numpy array with shape (`N_frames`, `N_animals`, 2) with the `xy` coordinate for each identity and frame in the video.
- ``version``: idtracker.ai version which created the current file.
- ``height``: Video height in pixels.
- ``width``: Video width in pixels.
- ``video_paths``: input video paths.
- ``frames_per_second``: input video frame rate.
- ``body_length``: mean body length computed as the mean value of the diagonal of all individual blob's bounding boxes.
- ``stats``: dictionary containing four different measurements of the session's identification accuracy.
- ``areas``: dictionary containing the mean, median and standard deviation of the blobs area for each individual.
- ``setup_points``: dictionary of the user defined setup points (from validator).
- ``identities_labels``: list of user defined identity labels (from validator).
- ``identities_groups``: list of user defined identity groups (from validator).
- ``id_probabilities``: Numpy array with shape (`N_frames`, `N_animals`) with the identity assignment certainty for each individual and frame of the video.
- ``length_unit``: ratio between the pixel distance and the real distance stated by the user of all pairs of points defined using the :ref:`length calibration` tool.
- ``silhouette_score``: Average silhouette score measured over a random sample of images at the end of the contrastive training.
- ``fragment_connectivity``: Connectivity of the fragments used in the contrastive step, computed as the average number of coexisting Fragments per Fragment, divided by the number of animals minus 1.

.. warning::
    ``body_length`` is not a reliable measurement of the real size of the animal. Its value depends on the segmentation parameters and video conditions.

Formats
=======

The compatible formats for trajectory files and how to load them in Python:

.. tip::
    You can convert the trajectory files of already tracked sessions to any of the compatible formats by running:

    .. code-block:: bash

        idtrackerai_format path/to/session_test --formats h5 npy csv pickle

HDF5
----

Hierarchical Data Formats from the :external:`HDF Group <https://www.hdfgroup.org/solutions/hdf5/>`.

- Binary format.
- Cross platform, readable by many languages and softwares.
- Becoming a standard in neuroscience.
- Requires an extra dependency to read/write in Python, :external:`HDF5 Python interface <https://docs.h5py.org/en/stable/>`.

.. code-block:: python

    import h5py # pip install h5py

    with h5py.File("./session_test/trajectories/trajectories.h5") as file:
        trajectories = file["trajectories"][:] # load trajectories into a Numpy file
        print(file.keys()) # check all Datasets and Groups in the file
        attributes = file.attrs # check other data stored in file's attributes

Numpy
-----

Numpy's :external:`binary serialization <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>` with :code:`np.save()`.

- Binary format.
- Only readable with Python.
- It uses Python's Pickle module when saving non-Numpy data.
- Not recommended for sharing since **The pickle module is not secure** (check `pickle documentation <https://docs.python.org/3/library/pickle.html>`_).
- This format is not ideal for this type of data but is maintained for legacy support.

.. code-block:: python

    import numpy as np

    trajectories_dict = np.load("./session_test/trajectories/trajectories.npy", allow_pickle=True).item()

Pickle
------

Python's :external:`Pickle module <https://docs.python.org/3/library/pickle.html>`.

- Binary format.
- Only readable with Python.
- Numpy uses it as the backend when saving non-Numpy data with ``np.save()``.
- Not recommended for sharing since **The pickle module is not secure** (check `pickle documentation <https://docs.python.org/3/library/pickle.html>`_).

.. code-block:: python

    import pickle

    with open("session_test/trajectories/trajectories.pickle", "rb") as file:
        trajectories_dict = pickle.load(file)


CSV and JSON
------------

- Human-readable format.
- Precision loss due to rounding.
- Universal and cross-platform.
- Spread over several files.

.. code-block:: python

    import json
    import numpy as np

    with open("session_test/trajectories/trajectories_csv/attributes.json", 'r') as file:
        attributes = json.load(file)

    # we skip the header (first row) and the time column
    trajectories = np.loadtxt("session_test/trajectories/trajectories_csv/trajectories.csv", skiprows=1, delimiter=",")[:, 1:]
    trajectories = trajectories.reshape(len(trajectories), -1, 2)
