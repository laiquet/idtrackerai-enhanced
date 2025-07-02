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

to launch the :ref:`segmentation app`, a graphical application designed to help you define the correct input parameters for your videos. There you can select the desired video(s) to track, set the basic parameters and start the tracking process.

Terminal usage
==============

From the :ref:`segmentation app`, you can start tracking directly or you can save the specified parameters in a *.toml* file like this one, enabling their reuse or automation in future tracking sessions:

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
    exclusive_roi = false


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

Like this, idtracker.ai can be run without graphical interface, directly from the terminal. This is useful for automation, batch processing or ssh remote tracking.

.. admonition:: Parameter log
  :class: sidebar warning

  Every loaded parameter will be notified in the :ref:`tracking log`, always read it while checking your parameters have been properly read.

More advanced parameters can be used to extend idtracker.ai's capabilities. These can be loaded from a *.toml* file by using the same ``--load`` argument (see the details of these :ref:`advanced parameters` below in this page).

Finally, any additional parameter can be passed in the command line as ``--PARAMETER VALUE``.

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

  Pay attention to your computer status during tracking (CPU, RAM and GPU usage). Idtracker.ai can be very memory-intensive in some parts (see :ref:`parallel processing`) and your computer can struggle on very long high resolution videos.

During tracking, idtracker.ai will communicate with the user through the log. This log will be displayed live in the terminal (Anaconda prompt on Windows) and written in the `idtrackerai.log` file in the current working directory. Users should keep an eye on the log, checking its status and warnings.

When a critical error occurs, the log contains all the information needed to solve it. Read the last lines of it to know more about what went wrong or send it to us at info@idtracker.ai so that we can help you. For more information on common errors after the installation, please refer to :ref:`installation troubleshooting`.

Advanced parameters
===================

Besides the basic parameters from the segmentation app (the ones in :ref:`example_toml`), more advanced parameters can be used.

.. important::

    - All parameter names are case-insensitive.
    - Define path variables using :toml:`'single quotes'` instead of :toml:`"double ones"` in the *toml* files to avoid backslashes (\\) to trigger special characters (see :external:`TOML documentation <https://toml.io>` to know more)
    - The value :toml:`''` in a *toml* file is loaded as a Python's :python:`None` in idtrackerai.

Output
------

- **OUTPUT_DIR.** Sets the directory path where the output session folder will be saved, by default it is the input video directory.

- **TRAJECTORIES_FORMATS.** The output trajectory files can be saved in four different formats: H5DF (:toml:`"h5"`), Numpy (:toml:`"npy"`), Python's pickle (:toml:`"pickle"`) and CSV (:toml:`"csv"`). Use this parameter to indicate the desired format(s) as a list of strings. Know more about these formats in :ref:`trajectory files`.

- **BOUNDING_BOX_IMAGES_IN_RAM** If true, bounding box images, a middle step to generate the final identification images, will be kept in RAM until no longer needed. Else, they are saved in disk and loaded when needed. We recommend setting this to :toml:`true` only when working with very slow disks (HDD) to speed up segmentation.

- **DATA_POLICY.** The tracking algorithm generates a significant amount of data, which is stored in the session folder along with the trajectory files. While some of this data can be large, it is also essential for running various additional tools included with the software.

  The available data policy options are: :toml:`"all"`, :toml:`"idmatcher.ai"`, :toml:`"knowledge_transfer"`, :toml:`"validation"` and :toml:`"trajectories"`.

  This setting determines which data is retained and, thus, which tools remain accessible. The default option, :toml:`"idmatcher.ai"`, provides a balanced approach between storage efficiency and tool availability.

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

The animal segmentation can be done by subtracting the background to each frame and thresholding this difference. To do this, a stack of sample frames is generated to later compute the background estimation using some statistical method.

- **BACKGROUND_SUBTRACTION_STAT.** Sets the statistic method to compute the background from the stack of sample frames, choices are :toml:`"median"` (default), :toml:`"mean"`, :toml:`"max"` (for dark animals on bright backgrounds) and :toml:`"min"` (for bright animals on dark backgrounds).

- **NUMBER_OF_FRAMES_FOR_BACKGROUND.** Sets the number of frames used to generate the stack of sample frames. These are equally spaced along the tracking intervals. More frames means more accuracy but also more computing time and RAM usage.

.. code-block:: toml
  :caption: Background subtraction defaults

  background_subtraction_stat = "median"
  number_of_frames_for_background = 50

Tracking checks
---------------

- **CHECK_SEGMENTATION.** The presence of frames with more blobs than animals means a bad segmentation with non-animal blobs detected. Idtracker.ai is not built to deal with non-animal blobs (shadows, reflections, dust...), these can contaminate the algorithms harming the identification. To ensure a proper segmentation, set this to :toml:`true` and idtracker.ai will abort the tracking session if a bad segmentation is detected.

  .. code-block:: toml

    check_segmentation = false

  .. note::
    This parameter appears on the segmentation app as :ref:`Stop tracking if #blobs > #animals`.


Parallel processing
-------------------

Some parts of idtracker.ai are parallelized (segmentation and identification images creation). This is done by slicing the video into different chunks and sending them to a group of independent workers to process.

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

- **KNOWLEDGE_TRANSFER_FOLDER.**: Sets the path to a *session* or *accumulation* folder from a previous tracked video. For example :toml:`"/home/username/session_test"` or :toml:`"/home/username/session_test/accumulation"`. This will load the weights of the models trained in the previous video as a starting point for the current session. It will also adopt the same **ID_IMAGE_SIZE** and **RESOLUTION_REDUCTION** as the previous video. By default, no knowledge is transferred and every identification model starts from scratch.

- **IDENTITY_TRANSFER.**: If the animals in your video are the same as the ones from the *knowledge_transfer* session, set this parameter to :toml:`true` to perform *identity transfer*. If so, idtracker.ai will use the network from the *knowledge_transfer* session to assign identities in the current session. In our experience, for this to work the video conditions need to be almost identical to the previous video.

- **ID_IMAGE_SIZE.** Identification images are squares, the size of which is, by default, optimized to match the size of the animals in each video. You can override this optimization by defining this parameter to an integer (the size in pixels of the side of the square images). Check the note below for more information about the behavior of this parameter.

- **RESOLUTION_REDUCTION.** Very big identification images (> 80 pixels per side) are usually unnecessarily heavy to work with. In this case, this parameter can scale down the images of the animals to fit them into smaller identification images, speeding up the tracking. It can go from 0 (limit to infinite reduction) to 1 (no reduction at all). Check the note below for more information about the behavior of this parameter.

.. note::

  The automatic values of **ID_IMAGE_SIZE** and **RESOLUTION_REDUCTION** are codependent in the following way:

  - If None of them are defined (default): the **ID_IMAGE_SIZE** is set based on the average size of the animals and the **RESOLUTION_REDUCTION** is used to limit this size to 80 pixels only if necessary.
  - Only **ID_IMAGE_SIZE** is defined by the user: only in case the animals average size is bigger than the stated image size, the resolution reduction is used to fit those animals in the images.
  - Only **RESOLUTION_REDUCTION** is defined by the user: the **ID_IMAGE_SIZE** is set based on the rescaled average size of the animals.

  We recommend to let idtrackerai define both parameters automatically, or to use the **KNOWLEDGE_TRANSFER_FOLDER** to inherit the parameters from a previously tracked video.

.. code-block:: toml
  :caption: Knowledge transfer defaults

  knowledge_transfer_folder = ''
  identity_transfer = false
  id_image_size = ''
  resolution_reduction = ''

.. tip::
    There are alternative ways of transferring identities between tracking sessions. Check our tool :ref:`idmatcher.ai`, it requires the identification image size and the resolution reduction factor to be equal for all the sessions.

Contrastive
-----------

Contrastive learning has been introduced in version 6.0.0 as the new identification algorithm (publication in progress). In it, all individual blobs are used to train :wikipedia:`ResNet <Residual_neural_network>` to embed images in an embedded space by using positive and negative pairs of images (this is why it's called contrastive learning). Positive pairs of images come from the same fragment and negative pairs come from different but coexisting fragments. With training, images from the same animal start clustering in the embedded space and their :wikipedia:`silhouette score <Silhouette_(clustering)>` increases reaching the target score. After contrastive training, images are embedded, clustered, identified and accumulated if possible. If enough images have been accumulated, the identification is completed, else the accumulation protocol starts by training the small idtrackerai's idCNN with the accumulated images from contrastive as a first synthetic global fragment.

- **DISABLE_CONTRASTIVE.** Skips the contrastive step to go directly to accumulation protocol.

- **CONTRASTIVE_MIN_ACCUMULATION.** Minimum fraction of images that need to be accumulated for taking the contrastive step as sufficient. If the fraction of accumulated images is lower than this value, the accumulation protocol will be run.

- **CONTRASTIVE_BATCHSIZE.** Number of pairs of images contained in a contrastive training batch. The more pairs of images, the more GPU memory will be needed. A batch of size :math:`N` will contain :math:`N` positive and :math:`N` negative pairs of images, so :math:`4N` images in total.

- **CONTRASTIVE_SILHOUETTE_TARGET.** Minimum silhouette score required for contrastive to finish training. This score, since coming from a K-Means clustering, is ranged from zero to one. Set it to one (unachievable value) to maximize the quality of the contrastive model and stopping the training only because of the triggering of the patience.

- **CONTRASTIVE_PATIENCE.** Number of steps without an improvement on the silhouette score to trigger the patience and early stopping the training during contrastive learning.

.. code-block:: toml
  :caption: Contrastive defaults

  disable_contrastive = false
  contrastive_min_accumulation = 0.5
  contrastive_batchsize = 400
  contrastive_silhouette_target = 0.91
  contrastive_patience = 30


Basic parameters
----------------

The assignment of any *basic* parameter (like the ones in :ref:`example_toml`) in the settings file acts as a default. For example, if you always track videos with 8 animals, you can set :toml:`number_of_animals = 8` in you settings file and, when running ``idtrackerai --load settings.toml``, the segmentation app will run with 8 animals as default.

Advanced hyper-parameters
-------------------------

.. warning:: These parameters change the way the CNN is trained, use with care.

- **THRESHOLD_EARLY_STOP_ACCUMULATION.**: Fraction of accumulated images needed to early stop the accumulation process.

- **MAXIMAL_IMAGES_PER_ANIMAL.**: Maximum number of images per animal that will be used to train the CNN in each accumulation step.

- **DEVICE.**: Device name passed to ``torch.device()`` to indicate where to perform machine learning operations, typically :toml:`"cpu"`, :toml:`"cuda"`, :toml:`"cuda:0"`... See :external:`Torch documentation <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`. (default: empty string, automatic device selection).

- **TORCH_COMPILE**. If set to :toml:`true`, all models will be compiled with :external:`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`. This can make the software run faster but may not be compatible with all devices. It's especially recommended for modern NVIDIA GPUs (H100, A100, or V100).

.. code-block:: toml

  threshold_early_stop_accumulation = 0.999
  maximal_images_per_animal = 3000
  device = ""
  torch_compile = false

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
    track_wo_identities = false
    roi_list = []

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
    identity_transfer = false
    id_image_size = ''
    resolution_reduction = ''

    # Contrastive
    disable_contrastive = false
    contrastive_min_accumulation = 0.5
    contrastive_batchsize = 400
    contrastive_silhouette_target = 0.91
    contrastive_patience = 30

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
