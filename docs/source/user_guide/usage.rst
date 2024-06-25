:sd_hide_title:

.. role:: toml(code)
   :language: toml

.. role:: python(code)
   :language: python

*****
Usage
*****

Basic Usage
===========

With idtracker.ai's Conda environment activated (activate it with ``conda activate idtrackerai``), run the command:

.. code:: bash

    idtrackerai

to launch the :ref:`segmentation app`, a graphical application designed to help you define the correct input parameters for your videos. There you can select the desired video(s) to track, set the basic parameters and start the tracking process.

Terminal usage
==============

From the :ref:`segmentation app`, you can start tracking directly or you can save the specified parameters in a *.toml* file like this one:

.. code-block:: toml
    :caption: example.toml
    :name: example_toml

    name = 'example'
    video_paths = ['/home/user/idtrackerai/video_A.avi']
    intensity_ths = [0, 155]
    area_ths = [100.0, inf]
    tracking_intervals = ""
    number_of_animals = 8
    use_bkg = false
    check_segmentation = false
    resolution_reduction = 1.0
    track_wo_identities = false
    roi_list = ['+ Polygon [[138.0, 50.1], [992.9, 62.1], [996.9, 878.9]]']
    exclusive_roi = false


This file contains the full configuration defined in the :ref:`segmentation app` and it can be loaded anytime with

.. code:: bash

    idtrackerai --load example.toml

to recover the app as you left it, or with

.. raw:: html

  <div class=highlight>

.. parsed-literal::

  idtrackerai --load example.toml **--track**

.. raw:: html

  </div>

to load the parameters from ``example.toml`` and **start the tracking process** without any prior graphical interface. This feature allows the control of idtracker.ai in remote via *ssh* and the capability to write custom scripts to run sequences of tracking sessions.

.. admonition:: Parameter log
  :class: sidebar warning

  Every loaded parameter will be notified in the :ref:`tracking log`, always read it while checking your parameters have been properly read.

More advanced parameters can be used to extend idtracker.ai's capabilities. These can be loaded from a *.toml* file by using the same ``--load`` argument (see the details of these :ref:`advanced parameters` below in this page).

Finally, any additional parameter can be passed in the command line as ``--PARAMETER VALUE``.

An example of an advanced idtracker.ai command could be:

.. code-block:: bash

    idtrackerai --load my_basic_settings.toml example.toml --track_wo_identities true --number_of_animals 15 --track

.. note::
    Parameters files defined in ``--load`` are processed in increasing order of priority, this means that the last ones override the ones before them. In the example above, declarations in :toml:`example.toml` would override the ones in :toml:`my_basic_settings.toml` (in case they are about the same parameter). Any command line declaration overrides all files from ``--load``.

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

  Pay attention to your computer status during tracking (CPU, RAM and GPU usage). Idtracker.ai can be very memory expensive in some parts (see :ref:`parallel processing`) and your computer can struggle on very long high resolution videos.

During tracking, idtracker.ai will communicate with the user through the log. This log will be displayed live in the terminal (Anaconda prompt on Windows) and written in the `idtrackerai.log` file in the current working directory. Users should keep an eye to the log, checking its status and warnings.

Idtracker may ask you to decide what to do if Protocol 3 is reached, see :ref:`tracking checks`.

When a critical error occurs, the log contains all the information to solve it. Read the last lines of it to know more about what went wrong or send it to us so that we can help you.

Advanced parameters
===================

Besides the basic parameters from the segmentation app (the ones in :ref:`example_toml`), more advanced parameters can be used.

.. note::

    - All parameters names are case insensitive.
    - Define path variables using :toml:`'single quotes'` instead of :toml:`"double ones"` in the *toml* files to avoid backslashes (\\) to trigger special characters (see :external:`TOML documentation <https://toml.io>` to know more)
    - The value :toml:`''` in a *toml* file is loaded as a Python's :python:`None`.

Output
------

- **OUTPUT_DIR.** Sets the directory path where the output session folder will be saved, by default it is the input video directory.

  .. code-block:: toml

    output_dir = ''

- **CONVERT_TRAJECTORIES_TO_CSV_AND_JSON.** The output trajectories are saved in a *.npy* file format (see :ref:`trajectory files`). This type of files are not human readable and can only be loaded with Python. If :toml:`true`, a copy of these files in *.csv* and *.json* formats are generated when running idtracker.ai. Since version 5.1.7, the default is True:

  .. code-block:: toml

    convert_trajectories_to_csv_and_json = true

- **BOUNDING_BOX_IMAGES_IN_RAM** If true, bounding box images, a middle step to generate the identification images, will be kept in RAM until no longer needed. Else they are saved in disk and loaded when needed. Set this to :toml:`true` when working with very slow disks (HDD) to speed up segmentation.

  .. code-block:: toml

    bounding_box_images_in_ram = false

- **ADD_TIME_COLUMN_TO_CSV.** If :toml:`true` and also :toml:`convert_trajectories_to_csv_and_json = true` a time column (in seconds) is added to the csv trajectory files, the default is :toml:`false`:

  .. code-block:: toml

    add_time_column_to_csv = false

- **DATA_POLICY.** The tracking algorithms generate lots of data saved in the session folder and some can be safely removed. Select one of the following policies to clean the output data when the tracking succeeds (ordered from less to more data expensive).

  - :toml:`"trajectories"`: only the trajectories will be saved, the rest of the data will be deleted.
  - :toml:`"validation"`: only the data necessary to validate the trajectories will be saved, the rest will be deleted.
  - :toml:`"knowledge_transfer"`: the data necessary to perform transfer learning or identity transfer will be kept.
  - :toml:`"idmatcher.ai"`: the data necessary to perform the matching of identities using :ref:`idmatcher.ai` will be kept. This option is the **optimal** one, removing only no longer needed data.
  - :toml:`"all"`: all the data generated during the tracking process is conserved (the default).

  .. code-block:: toml

    data_policy = "idmatcher.ai"

  .. tip::
    :toml:`data_policy = "idmatcher.ai"` is the optimal choice. It will delete only the data not going to be used in any case.

Background subtraction
----------------------

When subtracting background, a stack of video frames is generated to later compute the background estimation using some statist method

- **BACKGROUND_SUBTRACTION_STAT.** Sets the statistic method to compute the background, choices are :toml:`"median"` (default), :toml:`"mean"`, :toml:`"max"` (for dark animals on bright backgrounds) and :toml:`"min"` (for bright animals on dark backgrounds).

  .. code-block:: toml

    background_subtraction_stat = "median"

- **NUMBER_OF_FRAMES_FOR_BACKGROUND.** Sets the number of frames used to compute the background. These are equally spaced along the tracking intervals. More frames means more accuracy but also more computing time and RAM usage (only when computing the background).

  .. code-block:: toml

    number_of_frames_for_background = 50

Tracking checks
---------------

- **CHECK_SEGMENTATION.** The presence in the video of frames with more blobs than animals means a bad segmentation with non-animal blobs detected. Idtracker.ai is not built to deal with non-animal blobs (noise blobs), these can contaminate the algorithms making the identification harder. To ensure a proper segmentation, set this to :toml:`true` and idtracker.ai will abort the tracking session if a bad segmentation is detected.

  .. code-block:: toml

    check_segmentation = false

  .. note::
    This parameter appears on the segmentation app as :ref:`Stop tracking if #blobs > #animals`.

- **PROTOCOL3_ACTION.** Protocol 3 is called when both protocols 1 and 2 fail identifying animals. This protocol is **very** time consuming and, in most cases, it can be avoided by redefining the segmentation parameters. With this parameter you can choose the action idtracker.ai should take when facing Protocol 3. Choices are :toml:`"ask"` (ask the user to decide what to do by answering through the terminal), :toml:`"continue"` and :toml:`"abort"`. Default is :toml:`"ask"`.

  .. code-block:: toml

    protocol3_action = "ask"

Parallel processing
-------------------

Some parts in idtracker.ai are parallelized (segmentation and identification images creation). This is done by slicing the video in different chunks and giving them to a group of independent workers to process.

- **NUMBER_OF_PARALLEL_WORKERS.** Sets the number of workers used in the parallel parts. A negative value means using as many workers as the total number of CPUs minus the specified value. Zero value means running half of the total number of CPUs in the system or 8 if the system has more than 16 cores (using more than 8 cores doesn't provide any significant speed up). One means no multiprocessing at all. The default value is 0.

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

- **KNOWLEDGE_TRANSFER_FOLDER.**: Set the path to a *session* or *accumulation* folder from a previous tracked video. For example :toml:`"/home/username/session_test"` or :toml:`"/home/username/session_test/accumulation_0"`. By default, every identification protocol starts from scratch.

  .. code-block:: toml

    knowledge_transfer_folder = ''

- **IDENTITY_TRANSFER.**: If the animals being tracked are the same as the ones from the *knowledge_transfer* session, there is the possibility to perform *identity transfer*. If so, idtracker.ai will use the network from the *knowledge_transfer** session to assign the identities of the current session. In our experience, for this to work the video conditions need to be almost identical to the previous video. The default:

  .. code-block:: toml

    identity_transfer = false

- **ID_IMAGE_SIZE.** By default, identification images size are optimized for current animal sizes in each video. Override this behavior by defining this parameter to an integer (the size in pixels of the side of the square image). Useful to make sure two sessions have the same identification image size (used in :ref:`idmatcher.ai`)

  .. code-block:: toml

    id_image_size = ''

.. tip::
    There are alternative ways of transferring identities between tracking sessions. Check our tool :ref:`idmatcher.ai`, it requires the identification image size to be equal for all the sessions.

Basic parameters
----------------

The assignment of any *basic* parameter (like the ones in :ref:`example_toml`) in the settings file acts as a default. For example, if you always track videos with 8 animals, you can set :toml:`number_of_animals = 8` in you settings file and, when running ``idtrackerai --load settings.toml``, the segmentation app will run with 8 animals as default.

Advanced hyper-parameters
-------------------------

.. warning:: These parameters change the way the CNN is trained, use with care.

- **THRESHOLD_EARLY_STOP_ACCUMULATION.**: Ratio of accumulated images needed to early stop the accumulation process. By default:

  .. code-block:: toml

    threshold_early_stop_accumulation = 0.999

- **THRESHOLD_ACCEPTABLE_ACCUMULATION.**: Minimum ratio of accumulated images that an accumulation process needs to have at the end to be accepted as successful. By default:

  .. code-block:: toml

    threshold_acceptable_accumulation = 0.9

- **MAXIMAL_IMAGES_PER_ANIMAL.**: Maximum number of images per animal that will be used to train the CNN in each accumulation step. By default:

  .. code-block:: toml

    maximal_images_per_animal = 3000

- **DEVICE.**: Device name passed to ``torch.device()`` to indicate where machine learning computations will be performed, typically :toml:`"cpu"`, :toml:`"cuda"`, :toml:`"cuda:0"`... See :external:`Torch documentation <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`. (default: empty string, automatic device selection).

  .. code-block:: toml

    device = ""

File example
------------

An example settings file with all parameters as default (no effect) is

.. code-block:: toml
    :caption: example settings.toml

    # Segmentation app defaults
    name = ''
    video_paths = ''
    intensity_ths = [0, 130]
    area_ths = [50.0, inf]
    tracking_intervals = ""
    number_of_animals = 0
    use_bkg = false
    check_segmentation = false
    resolution_reduction = 1.0
    track_wo_identities = false
    roi_list = []

    # Output
    output_dir = ''
    convert_trajectories_to_csv_and_json = true
    bounding_box_images_on_ram = false
    add_time_column_to_csv = false
    data_policy = 'idmatcher.ai'

    # Background subtraction
    background_subtraction_stat = 'median'
    number_of_frames_for_background = 50

    # Parallel processing
    number_of_parallel_workers = 0
    frames_per_episode = 500

    # Knowledge and identity transfer
    knowledge_transfer_folder = ''
    identity_transfer = false
    id_image_size = ''

    # Tracking checks
    protocol3_action = "ask"

    # Advanced hyper-parameters
    threshold_early_stop_accumulation = 0.999
    threshold_acceptable_accumulation = 0.9
    maximal_images_per_animal = 3000
    device= ""

Complete list of idtracker.ai parameters
========================================

``idtrackerai -h`` prints the list of all possible command line arguments:

.. idtrackerai_argparser::
