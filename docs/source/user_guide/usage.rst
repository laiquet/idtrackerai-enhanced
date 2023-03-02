:sd_hide_title:

*****
Usage
*****

Basic Usage
===========

With idtracker.ai's Conda environment activated (``conda activate idtrackerai``), run the command:

.. code:: bash

    idtrackerai

to run the segmentation application, a graphical app designed to help you define the correct input parameters for your videos (more information about the :ref:`segmentation app`). From there, you can start tracking directly or you can save the specified parameters in a *.toml* file like this example:

.. code-block:: toml
    :caption: example.toml
    :name: example_toml

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


This file contains the full configuration defined in the segmentation app. It can be loaded anytime with

.. code:: bash

    idtrackerai --load example.toml

to recover the app as you left it or with

.. raw:: html

  <div class=highlight>

.. parsed-literal::

  idtrackerai --load example.toml **--track**

.. raw:: html

  </div>

to load the parameters from ``example.toml`` and **start the tracking process** without any prior graphical interface. This feature allows the control of idtracker.ai in remote via *ssh* and concede the capability to write custom scripts to run sequences of tracking sessions.

.. admonition:: Parameter log
  :class: sidebar warning
  
  Every loaded parameter will be notified in the :ref:`tracking log`, always read it checking your parameters have been properly read.

More advanced parameters can be used to extend idtracker.ai's capabilities. These can be loaded from a settings *.toml* file by using the ``--settings`` command argument (see the details of these :ref:`advanced parameters` below in this page).

Finally, any additional parameter can be passed in the command line as ``-PARAMETER VALUE``.

An example of an advanced idtracker.ai command could be:

.. code-block:: bash
    
    idtrackerai --settings my_basic_settings.toml --load example.toml --track_wo_identities true --number_of_animals 15 --track

.. note:: 
    Parameters definitions using ``--load`` method override the ones from ``--settings`` and any command line declaration overrides both input file methods.

.. tip::
  In the case of running idtracker.ai in remote (where the session parameters may have been created in another computer), it could be helpful to override, for example, the video paths from *example.toml*:


  .. raw:: html

    <div class=highlight>

  .. parsed-literal::

    idtrackerai --load example.toml **--video_paths path/in/remote/computer.avi** --track

  .. raw:: html

    </div>

Tracking log
============

.. admonition:: Take care of your machine
  :class: sidebar warning
    
  Pay attention to your computer status during tracking (CPU, RAM and GPU usage). Idtracker.ai can be very memory expensive on some parts (see :ref:`parallel processing`) and your computer can struggle in very long high resolution videos.

During tracking, idtracker.ai will communicate with the user through the log. This log will be live displayed in the terminal (Anaconda prompt on Windows) and written in the `idtrackerai.log` file in the current working directory. Users should keep an eye to the log checking its status and warnings.

When a critical error occur, the log contains all the information to solve it. Read the last lines of it to know more about what went wrong or send it to us so that we can help you.

Advanced parameters
===================

Besides the basic parameters from the segmentation app (the ones in :ref:`example_toml`), more advanced parameters can be used.

.. note:: 
    All parameters names are case insensitive.

Output
------

- **OUTPUT_DIR.** Sets the directory path where the output session folder will be saved, by default it is the input video directory.

  .. code-block:: toml

    output_dir = ''

- **CONVERT_TRAJECTORIES_TO_CSV_AND_JSON.** The output trajectories are saved in a *.npy* file format (see :ref:`trajectory files`). This type of files are not human readable and can only be loaded with Python. To generate a copy of these files in *.csv* and *.json* formats when running idtracker.ai set this parameter to ``true``. By default:

  .. code-block:: toml

    convert_trajectories_to_csv_and_json = false

- **DATA_POLICY.** The tracking algorithms generate lots of data saved in the session folder and some can be safely removed. Select one of the following policies to clean the output data when the tracking succeeds (ordered from less to more data expensive).

  - ``"trajectories"``: only the trajectories will be saved, the rest of the data will be deleted. 
  - ``"validation"``: only the data necessary to validate the trajectories will be saved, the rest will be deleted.
  - ``"knowledge_transfer"``: the data necessary to perform transfer learning or identity transfer will be kept.
  - ``"idmatcher.ai"``: the data necessary to perform the matching of identities using :ref:`idmatcher.ai` will be kept.
  - ``"all"``: all the data generated during the tracking process will be stored (the default).

  .. code-block:: toml

    data_policy = 'all'

Background subtraction
----------------------

When subtracting background, a stack of video frames is generated to later compute the background estimation using some statist method

- **BACKGROUND_SUBTRACTION_STAT.** Sets the statistic method to compute the background, choices are `median` (default), `mean`, `max` (for dark animals on bright backgrounds) and `min` (for bright animals on dark backgrounds).

  .. code-block:: toml

    background_subtraction_stat = 'median'

- **NUMBER_OF_FRAMES_FOR_BACKGROUND.** Sets the number of frames used to compute the background. These are equally spaced along the tracking intervals. More frames means more accuracy but also more computing time and RAM usage (only when computing the background).

  .. code-block:: toml

    number_of_frames_for_background = 50

Parallel processing
-------------------

Some parts in idtracker.ai are parallelized (segmentation and identification images creation). This is done by slicing the video in different chunks and giving them to a group of independent workers to process.

- **NUMBER_OF_PARALLEL_WORKERS.** Sets the number of workers used in the parallel parts. A negative value means using as many workers as the total number of CPUs minus the specified value. Zero value means running half of the total number of CPUs in the system. The default value is 0.

  .. code-block:: toml

    number_of_parallel_workers = 0

  .. warning:: 

    During segmentation, every worker can use up to 4GB of memory, using too many workers might fill your RAM memory very fast. Computers with a large number of CPU cores (>10) should be monitored and the number of parallel workers should be adjusted accordingly. 

- **FRAMES_PER_EPISODE.** Sets the size of the video chunks (episodes). Lass frames per episode means more parallel chunks. By default: 

  .. code-block:: toml

    frames_per_episode = 500

Knowledge and identity transfer
-------------------------------

You can use the knowledge acquired by a previously trained convolutional neural network as a starting point for the training and identification protocol. This can be useful to speed up the identification when the videos are **very** similar (same light conditions, same distance from camera to arena, same type and size of animals).

- **KNOWLEDGE_TRANSFER_FOLDER**: Set the path to an *accumulation* folder from a previous tracked session. For example `/home/username/session_test/accumulation_0`. By default, every identification protocol starts from scratch.

  .. code-block:: toml

    knowledge_transfer_folder = ''

- **IDENTITY_TRANSFER**: If the animals being tracked are the same as the ones from the *knowledge_transfer* session, there is the possibility to perform *identity transfer*. If so, idtracker.ai will use the network from the *knowledge_transfer** session to assign the identities of the current session. In our experience, for this to work the video conditions need to be almost identical to the previous video. The default:

  .. code-block:: toml

    identity_transfer = false

- **IDENTIFICATION_IMAGE_SIZE.** By default, identification images size are optimized for current animal sizes in each video. Override this behavior by defining this parameter to an integer (the size in pixels of the side of the square image). Useful to make sure two sessions have the same identification image size (used in :ref:`idmatcher.ai`)

  .. code-block:: toml

    identification_image_size = ''

.. note:: 
    There are alternative ways of transferring identities between tracking sessions. Check our tool :ref:`idmatcher.ai`, it requires the identification image size to be equal for all the sessions.


Basic parameters
----------------

The assignment of any *basic* parameter (like the ones in :ref:`example_toml`) in the settings file acts as a default. For example, if you always track videos with 8 animals, you can set ``number_of_animals = 8`` in you settings file and, when running ``idtrackerai --settings settings.toml``, the segmentation app will run with 8 animals as default.

File example
------------

An example settings file with all parameters as default (no effect) is

.. code-block:: toml
    :caption: example settings.toml

    # Segmentation app defaults 
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
    number_of_parallel_workers = 0
    frames_per_episode = 500

    # Knowledge and identity transfer
    knowledge_transfer_folder = ''
    identity_transfer = false
    identification_image_size = ''



Complete list of idtracker.ai parameters
========================================

Running ``idtrackerai -h`` will print a complete list of all possible arguments like the one above:

  --load session_parameters
                      ``path`` Primary .toml file to load session parameters
  --settings general_settings
                      ``path`` Secondary .toml file to load general settings
  --track               
                      Track the video without launching the GUI
  --tracking_intervals 
                      ``list_of_lists_of_two_ints`` Tracking intervals in frames. Examples: '[0,100]', '[[0,100],[150,200],...]'. If none, the whole video is tracked (default: None)
  --identity_transfer   ``bool`` If true, identities from knowledge transfer folder are transferred (default: False)
  --intensity_ths       ``list_of_two_ints`` Pixel's intensity thresholds
  --area_ths            ``list_of_two_ints`` Blob's areas thresholds
  --number_of_animals   ``int`` Number of different animals that appear in the video
  --output_dir          ``path`` Output directory where session folder will be saved to, default is video paths parent directory (default: None)
  --resolution_reduction 
                      ``float`` Video resolution reduction ratio (default: 1.0)
  --check_segmentation 
                      ``bool`` Check all frames have less or equal number of blobs than animals (default: False)
  --roi_list            ``str`` List of polygons defining the Region Of Interest (default: None)
  --use_bkg             ``bool`` Compute and extract background to improve blob identification (default: False)
  --video_paths  [ ...]
                      ``str`` List of paths to the video files to track
  --session             ``str`` Name of the session
  --track_wo_identities 
                      ``bool`` Track the video ignoring identities (without AI) (default: False)
  --convert_trajectories_to_csv_and_json 
                      ``bool`` If true, trajectories files are gonna be copied to .csv and .json files (default: False)
  --frames_per_episode 
                      ``int`` Maximum number of frames for each video episode (used to parallelize some processes) (default: 500)
  --knowledge_transfer_folder 
                      ``path`` Path to the session to transfer knowledge from (default: None)
  --background_subtraction_stat 
                      ``str`` Statistical method to compute the background (choices: median, mean, max, min) (default: median)
  --number_of_frames_for_background 
                      ``int`` Number of frames used to compute the background (default: 50)
  --number_of_parallel_workers 
                      ``int`` Maximum number of jobs to parallelize segmentation and identification image creation. A negative value means using the number of CPUs in the system minus the specified value. Zero means using half of the number of CPUs in the system (default: 0)
  --data_policy       
                      ``str`` Type of data policy indicating the data in the session folder not to be erased when successfully finished a tracking (choices: trajectories, validation, knowledge_transfer, idmatcher.ai, all) (default: all)
  --identification_image_size 
                      ``int`` The size of the identification images used in the tracking (default: -1)