****************
Output structure
****************

Idtracker.ai will generate a ``session_[SESSION_NAME]`` folder in the same directory where the input videos are (or in the ``--output_dir`` path if specified, see :ref:`output`). The session folder may be have the following structure:

.. admonition:: Note
    :class: sidebar note
    
    The content of the session folder may change depending on the necessities of each session. Also, ``--data_policy`` can remove some of the data after finishing a successful tracking (see :ref:`output`).

.. code-block:: 
    :caption: idtracker.ai session's output structure
    :emphasize-lines: 14, 15

    session_[SESSION_NAME]
    ├─ accumulation_*
    │  └─ ...
    ├─ crossings_detector
    │  └─ ...
    ├─ identification_images
    │  └─ id_images_*.hdf5
    ├─ preprocessing
    │  ├─ list_of_blobs.pickle
    │  └─ list_of_fragments.pickle
    ├─ segmentation_data
    │  └─ episode_images_*.hdf5
    ├─ trajectories
    │  ├─ trajectories.npy
    │  └─ trajectories_wo_gaps.npy
    ├─ video_object.json
    └─ idtrackerai.log

The trajectory files are the ones highlighted above, they contain the most valuable data for the end user, the position of every animal in every video frame. See how to read them in :ref:`trajectory files`.

In the session folder there's a copy of the session log ``idtrackerai.log`` made at the end of the process (successful or not). This file contains information of the entire tracking process and is essential to debug and understand idtracker.ai (see :ref:`tracking log`).

The majority of the generated data is a byproduct of the tracking process and it is not meant to be read or used by the end user. Still, an intuition of the data content can be read as:

- ``accumulation_*`` contains the identification network model and parameters. It can be used to match identities with other sessions with :ref:`idmatcher.ai`.
- ``crossings_detector`` contains the individual/crossing classification network model and parameters.
- ``identification_images`` contains the images used for identification. This is, an image for every animal and every frame on the video.
- ``preprocessing`` contains the blobs, fragments and global fragments objects (in Python pickle format). Advanced users can dive into these objects to gather some extra information about the tracking. Also, the ROI and the computed background are saved here.
- ``segmentation_data`` contains the temporal image used to generate the final identification images.
- ``video_object.json`` contains basic properties of the video and the session in human readable *.json* format.
