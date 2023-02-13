********************
Running idtracker.ai
********************

Usage
=====

In the Conda environment where idtracker.ai is installed, running the command

.. code:: bash

    idtrackerai

will execute the segmentation app, an app designed to help you define the correct input parameters for your video (more information about the :doc:`/segmentation_app`). From there, you can directly run idtracker.ai or save the specified parameters in a *.toml* file like this one 

.. _segmentation parameter file:
.. code-block:: toml
    :caption: example.toml

    session = 'example'
    video_paths = ['/home/user/idtrackerai/video_A.avi']
    intensity_ths = [0, 155]
    area_ths = [100, 60000]
    tracking_intervals = ""
    number_of_animals = 8
    use_bkg = false
    check_segmentation = false
    resolution_reduction = 1.0
    track_wo_identities = false
    roi_list = ['+ Polygon [[138.0, 50.1], [992.9, 62.1], [996.9, 878.9]]']


This file can be loaded later with

.. code:: bash

    idtrackerai --load example.toml

to recover the segmentation app as you left it or with

.. parsed-literal::

    idtrackerai --load example.toml **--track**

to start the tracking process without any graphical interface (useful to run idtracker.ai in remote via *ssh* or to use scripts performing a sequence of tracking sessions)

.. warning:: 
    Always read the console log checking your parameters have been successfully read.

More advanced parameters can be defined to extend idtracker.ai's capabilities. These can be defined in a settings *.toml* file which can be loaded with the :code:`--settings` flag.

Finally, any parameter can be defined in the terminal command by :code:`-PARAMETER VALUE`.

.. note::
  In the case of running idtracker.ai in remote, it could be helpful to override, for example, the video paths in *example.toml*:

  .. parsed-literal::
    idtrackerai --load example.toml **--video_paths path/in/remote/computer.avi** --track


A complete example of an advanced idtracker.ai command could be:

.. code-block:: bash
    
    idtrackerai --settings my_basic_settings.toml --load example.toml --track_wo_identities true --number_of_animals 15 --track

.. note:: 
    The :code:`--load` parameters override the :code:`--settings` ones and any terminal declaration overrides both input file methods.


Advanced parameters
===================


Segmentation app defaults
-------------------------

The definition of any parameter from the `segmentation parameter file`_ in a settings file will act as a default value. For example, if you always track videos with 8 animals, you can set :code:`number_of_animals = 8` in you settings file. When running :code:`idtrackerai --settings settings.toml`, the segmentation app will run with 8 animals as default.

.. note:: 
    All parameters names are case insensitive.

Output
------

- **OUTPUT_DIR.** Set the directory path to save the output session folder, by default the same directory as the input video paths

  .. code-block:: toml

    output_dir = ''

- **CONVERT_TRAJECTORIES_TO_CSV_AND_JSON.** The output trajectories are saved in a *.npy* file format. This type of files are not human readable and can only be loaded using Python. To get a copy of the output in *.csv* and *.json* formats when running idtracker.ai set

  .. code-block:: toml

    convert_trajectories_to_csv_and_json = true

- **DATA_POLICY.** The tracking process generates lots of data in the session folder, select one of the following policies to remove some of this data when the tracking succeeds (ordered from less to more data expensive).
    - **trajectories**: only the trajectories will be saved, the rest of the data will be deleted. 
    - **validation**: only the data necessary to validate the trajectories will be saved, the rest will be deleted.
    - **knowledge_transfer**: the data necessary to perform transfer learning or identity transfer will be kept.
    - **idmatcher.ai**: the data necessary to perform the matching of identities using `idmatcher.ai <https://gitlab.com/polavieja_lab/idmatcherai>` will be kept.
    - **all**: all the data generated during the tracking process will be stored (the default).

  .. code-block:: toml

    data_policy = 'all'

Background subtraction
----------------------

When subtracting background, a stack of video frames is generated to compute the background estimation using some statist method

- **BACKGROUND_SUBTRACTION_STAT.** Sets the statistic method, choices are `median` (default), `mean`, `max` (for bright background videos) and `min` (for dark background videos)

  .. code-block:: toml

    background_subtraction_stat = 'median'

- **NUMBER_OF_FRAMES_FOR_BACKGROUND.** Sets the number of frames used to compute the background. These are equally spaced along the tracking interval. More frames means more accuracy while more computing time and RAM usage is needed to copmute the background.

  .. code-block:: toml

    number_of_frames_for_background = 50

Parallel processing
-------------------

- **NUMBER_OF_JOBS_FOR_SEGMENTATION.** Set the number of parallel processes used to segment the video. A negative value means running as many processes as the total number of CPUs minus the specified number. The default value is -2.

  .. code-block:: toml

    number_of_jobs_for_segmentation = -2

  .. warning:: 

    During segmentation, every job can use up to 2GB of memory, using to many cores might fill your RAM memory very fast. Computers with a large number of CPU cores (>10) need to limit the number of parallel jobs. 

- **NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES**: The value of this constant is directly passed to the parameter *n_jobs* of the `class Parallel <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_ from the *joblib* package during the background subtraction process. Please read the documentation to set a valid value. Note that using to many cores might fill your memory very fast. The default value is -2, according to the documentation, all CPUs but one will be used.

  .. code-block:: toml

    number_of_jobs_for_setting_id_images = -2

- **FRAMES_PER_EPISODE.** The input video is processed in parallel by dividing it into smaller episodes. The length of these episodes is 500 frames, it can be modified with 

  .. code-block:: toml

    frames_per_episode = 500

Knowledge and identity transfer
-------------------------------

You can use the knowledge acquired by a previously trained convolutional neural network as a starting point for the training and identification protocol. This can be useful to speed up the identification when the videos are **very** similar (same light conditions, same distance from camera to arena, same type and size of animals).

- **KNOWLEDGE_TRANSFER_FOLDER_IDCNN**: Set the path to an *accumulation* folder from a previous tracked session. For example `/home/username/Session_test/accumulation_0`. By default, every identification protocol starts from scratch.

  .. code-block:: toml

    knowledge_transfer_folder_idcnn = ''


- **IDENTITY_TRANSFER**: If the animals being tracked are the same as the ones from the *knowledge_transfer* session, there is the possibility to perform *identity transfer*. If so, idtracker.ai will use the network from the *knowledge_transfer** session to assign the identities of the current session. In our experience, for this to work the video conditions need to be almost identical to the previous video. The default is False.

  .. code-block:: toml

    identity_transfer = false

.. note:: 

    There are alternative ways of transferring or matching identities between videos. Check the tool `idmatcher.ai <https://gitlab.com/polavieja_lab/idmatcherai>`_. To use this tool, the size of the identification images needs to be the same for all the videos. In the future, idmatcher.ai project will be merged into idtracker.ai

- **IDENTIFICATION_IMAGE_SIZE.** by default, identification images size are optimized using current animals sizes in the video. Set this parameter to an integer (the size in pixels of one side of the square image) if you want to make sure two sessions have the same identification image size
  .. code-block:: toml

    identification_image_size = ''





File example
------------

A settings file with all parameters as default (no effect) is

.. code-block:: toml
    :caption: settings.toml

    # GUI defaults 
    session = ''
    video_paths = ''
    intensity_ths = [0, 155]
    area_ths = [100, 60000]
    tracking_intervals = ""
    number_of_animals = 0
    use_bkg = false
    check_segmentation = false
    resolution_reduction = 1.0
    track_wo_identities = false
    roi_list = []

    # Output
    output_dir = ''
    convert_trajectories_to_csv_and_json = false
    data_policy = 'all'

    # Background subtraction
    background_subtraction_stat = 'median'
    number_of_frames_for_background = 50

    # Parallel processing
    number_of_jobs_for_segmentation = -2
    number_of_jobs_for_setting_id_images = -2
    frames_per_episode = 500

    # Knowledge and identity transfer
    knowledge_transfer_folder_idcnn = ''
    identity_transfer = false
    identification_image_size = ''



Complete list of idtracker.ai parameters
========================================

Running :code:`idtrackerai -h` will print a complete list of all possible arguments like the one above:

--load session_parameters
                    (path) Primary .toml file to load session parameters
--settings general_settings
                    (path) Secondary .toml file to load general settings
--track               
                    Track the video without launching the GUI
--tracking_intervals 
                    (list_of_lists_of_two_ints) Tracking intervals in frames. Examples: '[0,100]', '[[0,100],[150,200],...]'. If none, the whole video is tracked (default: None)
--identity_transfer   (bool) If true, identities from knowledge transfer folder are transferred (default: False)
--intensity_ths       (list_of_two_ints) Pixel's intensity thresholds
--area_ths            (list_of_two_ints) Blob's areas thresholds
--number_of_animals   (int) Number of different animals that appear in the video
--output_dir          (path) Output directory where session folder will be saved to, default is video paths parent directory (default: None)
--resolution_reduction 
                    (float) Video resolution reduction ratio (default: 1.0)
--check_segmentation 
                    (bool) Check all frames have less or equal number of blobs than animals (default: False)
--roi_list            (str) List of polygons defining the Region Of Interest (default: None)
--use_bkg             (bool) Compute and extract background to improve blob identification (default: False)
--video_paths  [ ...]
                    (str) List of paths to the video files to track
--session             (str) Name of the session
--track_wo_identities 
                    (bool) Track the video ignoring identities (without AI) (default: False)
--convert_trajectories_to_csv_and_json 
                    (bool) If true, trajectories files are gonna be copied to .csv and .json files (default: False)
--frames_per_episode 
                    (int) Maximum number of frames for each video episode (used to parallelize some processes) (default: 500)
--knowledge_transfer_folder 
                    (path) Path to the session to transfer knowledge from (default: None)
--background_subtraction_stat 
                    (str) Statistical method to compute the background (choices: median, mean, max, min) (default: median)
--number_of_frames_for_background 
                    (int) Number of frames used to compute the background (default: 50)
--number_of_jobs_for_segmentation 
                    (int) Maximum number of jobs to parallelize segmentation (default: -2)
--data_policy         (str) Type of data policy indicating the data in the session folder not to beerased when successfully finished a tracking (choices:
                    trajectories, validation, knowledge_transfer, idmatcher.ai, all) (default: all)
--identification_image_size 
                    (int) The size of the identification images used in the tracking (default: -1)
--number_of_jobs_for_setting_id_images 
                    (int) Maximum number of jobs to parallelize identification images creation (default: -2)