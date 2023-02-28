****************
Output structure
****************

Idtracker.ai will generate a ``session_[SESSION_NAME]`` folder in the same directory where the input videos are (or in the ``--output_dir`` path if specified, see :ref:`advanced parameters<output>`). It may be have the structure below:

.. admonition:: Note
    :class: sidebar note
    
    The content of the session folder may change depending on the necessities of each session. Also, ``--data_policy`` can remove some of the data after finishing a successful tracking (see :ref:`advanced parameters<output>`).

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

The majority of the generated data is a byproduct of the tracking process and it is not meant to be read or used by the end user. Still, an intuition of the data content can be read as:

- ``accumulation_*`` contains the identification network parameters. It can be used to match identities with other sessions with idmatcher.ai.
- ``crossings_detector`` contains the individual/crossing classification network parameters.
- ``identification_images`` contains the images used for identification. This is, an image for every animal and every frame on the video.
- ``segmentation_data`` contains the temporal image before being processed to become identification images.
- ``video_object.json`` contains basic video properties in human readable *.json* format.
