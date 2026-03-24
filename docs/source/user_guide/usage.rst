:sd_hide_title:

.. role:: toml(code)
   :language: toml

.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

*****
Usage
*****

Basic Usage
===========

Activate idtracker.ai's Conda environment using ``conda activate idtrackerai``, then run the command:

.. code:: bash

    idtrackerai

to launch the :ref:`segmentation app`, a graphical application where you can select videos, set parameters, and start the tracking process.

Terminal usage
==============

From the :ref:`segmentation app`, you can either start tracking directly or save your parameters to a TOML file for reuse and automation:

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
    track_wo_identities = false
    roi_list = ['+ Polygon [[138.0, 50.1], [992.9, 62.1], [996.9, 878.9]]']
    exclusive_rois = false


This file contains the full configuration defined in the :ref:`segmentation app` and it can be loaded anytime with:

.. code:: bash

    idtrackerai --load example.toml

to recover the app as you left it. Add the ``--track`` flag to **start the tracking process** directly from the terminal:

.. raw:: html

  <div class=highlight>

.. parsed-literal::

  idtrackerai --load example.toml **--track**

.. raw:: html

  </div>

This way, idtracker.ai can be run without a graphical interface, directly from the terminal — useful for automation, batch processing, or remote SSH tracking.

.. admonition:: Parameter log
  :class: sidebar warning

  Every loaded parameter will be notified in the :ref:`tracking log`, always read it while checking your parameters have been properly read.

Advanced parameters can also be loaded from a *.toml* file using the same ``--load`` argument (see :ref:`advanced parameters` below).

Any parameter can also be passed directly on the command line as ``--PARAMETER VALUE``.

An example of an advanced idtracker.ai command could be:

.. code-block:: bash

    idtrackerai --load my_basic_settings.toml example.toml --track_wo_identities true --number_of_animals 15 --track

.. note::
    Parameter files specified with ``--load`` are processed in the order they are listed, with later files overriding earlier ones for the same parameter. For example, in the command above, settings in :toml:`example.toml` will take precedence over those in :toml:`my_basic_settings.toml`. Additionally, any parameter specified directly on the command line will override all settings from the loaded files.

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

  Pay attention to your computer status during tracking (CPU, RAM and GPU usage). idtracker.ai can be very memory-intensive in some parts (see :ref:`parallel processing`) and your computer can struggle on very long high resolution videos.

During tracking, idtracker.ai reports its progress through the log. It is displayed live in the terminal (Anaconda Prompt on Windows) and written to ``idtrackerai.log`` in the current working directory. Keep an eye on it — it reports loaded parameters, warnings, and the result of each tracking step.

When a critical error occurs, the last lines of the log describe what went wrong. You can send the log file to info@idtracker.ai for support. For common post-installation errors, see :ref:`installation troubleshooting`.

Advanced parameters
===================

In addition to the basic parameters shown in :ref:`example_toml`, idtracker.ai supports the following advanced parameters.

.. important::

    - All parameter names are case-insensitive.
    - Define path variables using :toml:`'single quotes'` instead of :toml:`"double ones"` in the *toml* files to avoid backslashes (\\) to trigger special characters (see :external:`TOML documentation <https://toml.io>` to know more)
    - The value :toml:`''` in a *toml* file is loaded as a Python's :python:`None` in idtracker.ai.

Output
------

- **OUTPUT_DIR.** Sets the directory path where the output session folder will be saved, by default it is the input video directory.

- **TRAJECTORIES_FORMATS.** The output trajectory files can be saved in four different formats: H5DF (:toml:`"h5"`), Numpy (:toml:`"npy"`), Python's pickle (:toml:`"pickle"`), CSV (:toml:`"csv"` or :toml:`"csv_tidy"`), and Parquet (:toml:`parquet`). Use this parameter to indicate the desired format(s) as a list of strings. Know more about these formats in :ref:`trajectory files`.

- **BOUNDING_BOX_IMAGES_IN_RAM** If true, bounding box images (an intermediate step in generating identification images) are kept in RAM until no longer needed. Otherwise, they are saved to disk and loaded when needed. Only set this to :toml:`true` when working with very slow disks (HDDs) to speed up segmentation.

- **DATA_POLICY.** The tracking algorithm stores a significant amount of intermediate data alongside the trajectory files. This setting controls how much of it is kept — retaining more enables more post-processing tools, while retaining less saves disk space.

  The available options are: :toml:`"all"`, :toml:`"idmatcher.ai"`, :toml:`"knowledge_transfer"`, :toml:`"validation"` and :toml:`"trajectories"`.

  The default, :toml:`"idmatcher.ai"`, balances storage efficiency with tool availability.

  The table below details which data is preserved under each policy and which tools remain functional.

.. _data_policy_data_table:

.. list-table:: Data availability for different data policies
  :header-rows: 1
  :stub-columns: 1

  * -
    - ``all``
    - ``idmatcher.ai``
    - ``knowledge_transfer``
    - ``validation``
    - ``trajectories``
  * - Trajectories
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ✅
  * - Pre-processing folder
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ❌
  * - Identification model folder
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ❌
    - .. centered:: ❌
  * - Identification images
    - .. centered:: ✅
    - .. centered:: ✅
    - .. centered:: ❌
    - .. centered:: ❌
    - .. centered:: ❌
  * - Crossing detector folder
    - .. centered:: ✅
    - .. centered:: ❌
    - .. centered:: ❌
    - .. centered:: ❌
    - .. centered:: ❌
  * - Bounding box images
    - .. centered:: ✅
    - .. centered:: ❌
    - .. centered:: ❌
    - .. centered:: ❌
    - .. centered:: ❌

.. _data_policy_tools_table:

.. list-table:: Tool availability for different data policies
   :header-rows: 1
   :stub-columns: 1

   * -
     - ``all``
     - ``idmatcher.ai``
     - ``knowledge_transfer``
     - ``validation``
     - ``trajectories``
   * - :ref:`Video Generator`
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ✅
   * - :ref:`validator_reference`
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ❌
   * - :ref:`knowledge transfer`
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ❌
     - .. centered:: ❌
   * - :ref:`idmatcher.ai`
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ❌
     - .. centered:: ❌
     - .. centered:: ❌
   * - :ref:`cluster inspection`
     - .. centered:: ✅
     - .. centered:: ✅
     - .. centered:: ❌
     - .. centered:: ❌
     - .. centered:: ❌

.. code-block:: toml
  :caption: Output defaults

  output_dir = ''
  trajectories_formats = ["h5", "npy", "csv"]
  bounding_box_images_in_ram = false
  data_policy = "idmatcher.ai"

Background subtraction
----------------------

Animal segmentation works by subtracting the background from each frame and applying a threshold to the difference image. A stack of sample frames is used to estimate the background with a statistical method.

- **BACKGROUND_SUBTRACTION_STAT.** Sets the statistical method used to compute the background from a sample of frames. Options are: :toml:`"median"` (default, works well in most cases), :toml:`"mean"` (default in version 4), :toml:`"max"` (recommended for dark animals on bright backgrounds), and :toml:`"min"` (for bright animals on dark backgrounds). You can also supply a file path to a pre-computed background image (PNG) instead of one of these keywords.

  .. tip::

    Pre-compute the background of a video with the command :code:`idtrackerai_background`. The resulting PNG can be passed directly to **BACKGROUND_SUBTRACTION_STAT**, avoiding recomputation on every tracking run.

- **NUMBER_OF_FRAMES_FOR_BACKGROUND.** Sets the number of frames used to generate the stack of sample frames. These are equally spaced along the tracking intervals. More frames means more accuracy but also more computing time and RAM usage.

.. code-block:: toml
  :caption: Background subtraction defaults

  background_subtraction_stat = "median"
  number_of_frames_for_background = 50

Tracking checks
---------------

- **CHECK_SEGMENTATION.** Frames containing more blobs than animals indicate a bad segmentation — non-animal blobs (shadows, reflections, dust, etc.) have been detected. These can contaminate the tracking algorithms and degrade identification accuracy. Set this to :toml:`true` to have idtracker.ai abort the session if a bad segmentation is detected.

  .. code-block:: toml

    check_segmentation = false

  .. note::
    This parameter appears on the segmentation app as :ref:`Stop tracking if #blobs > #animals`.


Parallel processing
-------------------

Some parts of idtracker.ai are parallelized — specifically, segmentation and identification image creation. The video is sliced into chunks, each processed by an independent worker.

- **NUMBER_OF_PARALLEL_WORKERS.** Sets the number of workers used in the parallel parts of the application.

  - A negative value indicates using as many workers as the total number of CPUs minus the specified value.
  - A value of zero means running half of the total number of CPUs in the system. If the system has more than 8 cores, defaults to 4 workers, as using more than 4 cores does not provide significant speed-up.
  - A positive value explicitly sets the number of workers to the specified value.
  - One means not using multiprocessing at all.

  The default value is 0.

  .. warning::

    During segmentation, every worker can use up to 4GB of memory, using too many workers might fill your RAM memory very fast. Computers with a large number of CPU cores (>10) should be monitored and the number of parallel workers should be adjusted accordingly. For users with limited RAM, consider reducing the number of parallel workers. Additionally, use monitoring tools like `htop`, `top`, or `free` on Linux, Task Manager or Resource Monitor on Windows, and Activity Monitor on macOS to keep an eye on your system's resource usage.

- **FRAMES_PER_EPISODE.** Sets the size of the video chunks (episodes). Less frames per episode means more parallel chunks.

.. code-block:: toml
  :caption: Parallel processing defaults

  number_of_parallel_workers = 0
  frames_per_episode = 500

Knowledge transfer
------------------

You can use the knowledge acquired by the identification model of a previous video as a starting point for the training of the current one. This speeds up the identification training when the videos are **very** similar (same light conditions, distance from camera to arena, type and size of animals).

- **KNOWLEDGE_TRANSFER_FOLDER.** Path to a previous *session* or *accumulation* folder whose trained model will be used to initialize the current session. Examples: :toml:`'/home/username/session_test'` or :toml:`'/home/username/session_test/accumulation'`.

  When set, idtracker.ai will:

  - load the model weights from the specified folder as a starting point for training the current identification model,
  - adopt the same **ID_IMAGE_SIZE** and **RESOLUTION_REDUCTION** values from the previous session,
  - attempt to transfer identities automatically when conditions are sufficiently similar (lighting, camera distance, animal size/appearance).

  .. note::

    - Video conditions must be nearly identical for reliable identity transfer; otherwise identities can be misassigned.
    - If the folder does not contain the expected trained model, idtracker.ai will log a warning and proceed without transfer.
    - By default (empty value), no knowledge transfer is performed and identification models start training from random weights with arbitrary identity assignments.

- **ID_IMAGE_SIZE.** Identification images are square; by default their size is set automatically to match the animals' size in the video. Set this to an integer to override the automatic size (pixels per side).

- **RESOLUTION_REDUCTION.** When animal images are large (> 80 pixels per side), this parameter scales them down to reduce computation. It ranges from 0 (maximum reduction) to 1 (no reduction).

.. note::

  When set automatically, **ID_IMAGE_SIZE** and **RESOLUTION_REDUCTION** interact as follows:

  - Neither defined (default): **ID_IMAGE_SIZE** is set from the average animal size; **RESOLUTION_REDUCTION** is applied only if the result would exceed 80 pixels.
  - Only **ID_IMAGE_SIZE** defined: resolution reduction is applied only if the average animal size exceeds the specified image size.
  - Only **RESOLUTION_REDUCTION** defined: **ID_IMAGE_SIZE** is set from the rescaled average animal size.

  We recommend letting idtracker.ai set both automatically, or using **KNOWLEDGE_TRANSFER_FOLDER** to inherit them from a previous session.

.. code-block:: toml
  :caption: Knowledge transfer defaults

  knowledge_transfer_folder = ''
  id_image_size = ''
  resolution_reduction = ''

.. tip::
    There are alternative ways of transferring identities between tracking sessions. Check our tool :ref:`idmatcher.ai`, it requires the identification image size and the resolution reduction factor to be equal for all the sessions.

Contrastive
-----------

Since version 6.0.0, idtracker.ai uses contrastive learning as its primary identification algorithm (described in the `2025 eLife publication <https://doi.org/10.7554/eLife.107602>`_). A :wikipedia:`ResNet <Residual_neural_network>` network is trained to map animal images into an embedding space where images of the same individual cluster together. Training uses *positive pairs* (images from the same fragment) and *negative pairs* (images from different, co-existing fragments). As training progresses, the :wikipedia:`silhouette score <Silhouette_(clustering)>` of the resulting clusters rises toward the target threshold.

Once the target silhouette score is reached (or the patience limit is hit), the embedded images are clustered and identities are assigned. If the accumulated fraction of identified images is sufficient, tracking is complete. Otherwise, the classical accumulation protocol takes over, using the contrastive results as a warm start for idtracker.ai's idCNN.

- **DISABLE_CONTRASTIVE.** Skips the contrastive step to go directly to accumulation protocol.

- **CONTRASTIVE_MIN_ACCUMULATION.** Minimum fraction of images that need to be accumulated for taking the contrastive step as sufficient. If the fraction of accumulated images is lower than this value, the accumulation protocol will be run.

- **CONTRASTIVE_BATCHSIZE.** Number of pairs of images contained in a contrastive training batch. The more pairs of images, the more GPU memory will be needed. A batch of size :math:`N` will contain :math:`N` positive and :math:`N` negative pairs of images, so :math:`4N` images in total.

- **CONTRASTIVE_SILHOUETTE_TARGET.** Minimum silhouette score required for contrastive to finish training. This score, since coming from a K-Means clustering, is ranged from zero to one. Set it to one (unachievable value) to maximize the quality of the contrastive model and stopping the training only because of the triggering of the patience.

- **CONTRASTIVE_PATIENCE.** Number of steps without an improvement on the silhouette score to trigger the patience and early stopping the training during contrastive learning.

- **CONTRASTIVE_MAX_MBYTES.** Maximum RAM (in megabytes) reserved for preloading identification images during contrastive training. By default (empty value), a quarter of the available system memory is used.

.. code-block:: toml
  :caption: Contrastive defaults

  disable_contrastive = false
  contrastive_min_accumulation = 0.5
  contrastive_batchsize = 400
  contrastive_silhouette_target = 0.91
  contrastive_patience = 30
  contrastive_max_mbytes = ''


Basic parameters
----------------

The assignment of any *basic* parameter (like the ones in :ref:`example_toml`) in the settings file acts as a default. For example, if you always track videos with 8 animals, you can set :toml:`number_of_animals = 8` in your settings file and, when running ``idtrackerai --load settings.toml``, the segmentation app will open with 8 animals as the default.

Advanced hyper-parameters
-------------------------

.. warning:: These parameters change the way the CNN is trained, use with care.

- **THRESHOLD_EARLY_STOP_ACCUMULATION.** Fraction of accumulated images needed to early-stop the accumulation process.

- **MAXIMAL_IMAGES_PER_ANIMAL.** Maximum number of images per animal used to train the CNN in each accumulation step.

- **DEVICE.** Device name passed to ``torch.device()`` to specify where machine learning operations run, typically :toml:`"cpu"`, :toml:`"cuda"`, :toml:`"cuda:0"`, etc. See :external:`Torch documentation <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`. (default: empty string, automatic device selection).

- **TORCH_COMPILE**. If set to :toml:`true`, all models will be compiled with :external:`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`. This can make the software run faster but may not be compatible with all devices.

.. code-block:: toml

  threshold_early_stop_accumulation = 0.999
  maximal_images_per_animal = 3000
  device = ""
  torch_compile = false

File example
------------

An example settings file with all parameters set to their defaults (no effect):

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
    track_wo_identities = false
    roi_list = ''
    exclusive_rois = false

    # Output
    output_dir = ''
    trajectories_formats = ["h5", "npy", "csv"]
    bounding_box_images_in_ram = false
    data_policy = 'idmatcher.ai'

    # Background subtraction
    background_subtraction_stat = 'median'
    number_of_frames_for_background = 50

    # Parallel processing
    number_of_parallel_workers = 0
    frames_per_episode = 500

    # Knowledge and identity transfer
    knowledge_transfer_folder = ''
    id_image_size = ''
    resolution_reduction = ''

    # Contrastive
    disable_contrastive = false
    contrastive_min_accumulation = 0.5
    contrastive_batchsize = 400
    contrastive_silhouette_target = 0.91
    contrastive_patience = 30
    contrastive_max_mbytes = ''

    # Advanced hyper-parameters
    threshold_early_stop_accumulation = 0.999
    maximal_images_per_animal = 3000
    device= ""
    torch_compile = false

``idtrackerai -h``
------------------

Use the command ``idtrackerai -h`` to print the list of all possible command line arguments in your terminal:

.. dropdown:: Output of ``idtrackerai -h``
    :animate: fade-in
    :icon: info
    :color: secondary

    .. idtrackerai_argparser::

Usage Analytics
===============

idtracker.ai collects usage analytics to improve the software. This data is anonymized, does not include any sensitive or personally identifiable information, and is collected in every command line call to an idtracker.ai module (not collected for API calls). It is used solely for research purposes and contains general information such as the operating system, the version of idtracker.ai being used, and the command used to run it.

You may check out the :external:`source code <https://gitlab.com/polavieja_lab/idtrackerai/-/blob/master/src/idtrackerai/utils/telemetry.py>` to see exactly what data is collected.

If you want to opt-out of this data collection, go to the *"About"* menu located in the top-left corner of any idtracker.ai graphical application and uncheck the box for analytics. Alternatively, set the environment variable ``IDTRACKERAI_DISABLE_ANALYTICS`` to ``true`` or ``1``. For example:

- On Linux/macOS: :bash:`export IDTRACKERAI_DISABLE_ANALYTICS=true`
- On Windows: :bash:`set IDTRACKERAI_DISABLE_ANALYTICS=true`

Any of these actions will keep analytics disabled until you enable them again.
