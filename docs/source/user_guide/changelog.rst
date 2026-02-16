*********
Changelog
*********

.. admonition:: Read about the new identification algorithm in :ref:`6.0.0` in our latest publication

    .. image:: ../_static/elife-white-horizontal-2020.svg
        :class: only-dark
        :target: https://doi.org/10.7554/eLife.107602
        :alt: idtracker.ai publication in eLife (2025)
        :width: 30%
        :align: left
        :height: 120

    .. image:: ../_static/elife-full-color-horizontal-2020.svg
        :class: only-light
        :target: https://doi.org/10.7554/eLife.107602
        :alt: idtracker.ai publication in eLife (2025)
        :width: 30%
        :align: left
        :height: 120

    .. centered:: `New idtracker.ai: rethinking multi-animal tracking as a representation learning problem to increase accuracy and reduce tracking times <https://doi.org/10.7554/eLife.107602>`_

Authors since :ref:`5.0.0`: Jordi Torrents (jordi.torrents@research.fchampalimaud.org | jordi.torrentsm@gmail.com)

Unreleased
==========

- Fix non working App icon in Linux taskbar (Ubuntu, Gnome).
- Fix bug when setting custom background in Segmentation App.
- Allow disabling impossible jumps detection in Validator.
- Handle ``PermissionError`` when tracking from a directory with no write permissions.

6.0.13
======

- Add the entrypoint ``idtrackerai_background`` to compute the background of a video or set of videos from terminal, see :ref:`background subtraction`.
- Remove parameter ``IDENTITY_TRANSFER`` setting it to ``True`` if a knowledge transfer folder is provided.
- Fix bug in the :ref:`validator_reference` when saving session from the pop up dialog at closing the GUI.
- Fix bug in the :ref:`validator_reference` when opening a session from the pop up dialog at starting the GUI.
- Add identities to fragments when tracking single animal or single fragment videos. This allows to use idmatcher.ai afterwards.
- Handle sessions without identification models and fragments without identities (when ``n_animals=1``) in :ref:`idmatcher.ai`.
- Add a "*save*" button in the Segmentation App to store the computed background as an image file.
- Do not set the automatic resolution reduction if the recommended value is too close to 1 (bigger than 0.91).
- Add Parquet format to the possible trajectory output formats, see :ref:`formats`.

6.0.12
======

- Fix https://gitlab.com/polavieja_lab/idtrackerai/-/issues/105 regarding custom background loading in Segmentation App.

6.0.11
======

- Fix https://gitlab.com/polavieja_lab/idtrackerai/-/issues/104 regarding the `forkserver` start method in MacOS.

6.0.10
======

- Add the possibility to upload a custom background in the Segmentation App and by using the parameter ``BACKGROUND_SUBTRACTION_STAT``, see :ref:`Background subtraction`.
- Ask for a session to open if no session directory is provided when opening the :ref:`validator_reference`.


6.0.8
=====

- Add color settings in the :ref:`validator_reference` and :ref:`video generator`.
- Fix incompatibility with older versions of OpenCV.
- Automatic pop ups in GUIs only when there is a major upgrade available (not in patch updates).
- Remove bias parameter from the last fully connected layer in ResNet models since only relative distances in the representation space matter.
- Redefined ``fragment_connectivity`` with a factor x2 and added its value to the :ref:`trajectory files`.
- Removed ``estimated_accuracy_identified`` and ``estimated_accuracy_after_interpolation`` from the :ref:`trajectory files`. The value of ``estimated_accuracy`` should be used instead to estimate the accuracy of the tracking.

6.0.7
=====

- Fix incompatibility with PyQt5.

6.0.6
=====

- Implement tracking without identities together with exclusive ROIs. This allows to track videos without keeping track of the identities but still respecting the exclusive ROIs defined in the segmentation app.
- Fix a bug in the :ref:`validator_reference` with ``numpy.bool`` getting into PyQt system.
- Improve main Argument Parser descriptions.
- Fix incompatibility with non-ASCII (chinese) characters in paths.
- Fix frozen Validator GUI when opening and saving sessions due to bad parallel processing management.
- Add new parameter ``CONTRASTIVE_MIN_ACCUMULATION`` to set the minimum fraction of images that need to be accumulated for taking the contrastive step as sufficient. If the fraction of accumulated images is lower than this value, the accumulation protocol will be run. This parameter is set to 0.5 by default.
- Optimized object rendering in the :ref:`validator_reference` preview by checking if the object is visible in the current frame before rendering it.
- Fix video player not respecting original video framerate.

6.0.5
=====

- Add a new GUI for the :ref:`video generator`.
- Better handling of errors in GUIs.
- Fix incompatibility with latest version of superqt https://gitlab.com/polavieja_lab/idtrackerai/-/issues/102.

6.0.3
=====

- Fix a critical bug in the Validator that was causing the GUI to crash when exploring the last frames of the video.
- Fix documentation issues.

6.0.2
=====

- Hotfix for a critical ``AttributeError`` at ``compute_identity_probabilities``.

6.0.1
=====

- Hotfix for non responding GUI issues on MacOS.

6.0.0
=====

- The Protocol 3 (used as a fallback of Protocol 2) has been removed and Contrastive Protocol has been introduced as the new main identification protocol leaving Accumulation Protocol (Protocol 2) as a fallback to run if Contrastive fails (publication in progress).
- Trajectories can be saved in a variety of formats following the new parameter ``TRAJECTORIES_FORMATS`` (:ref:`check the docs <output>`). As a consequence, the parameter ``CONVERT_TRAJECTORIES_TO_CSV_AND_JSON`` has been removed.
- Resolution reduction has been removed from the Segmentation App and it has now a default automatic value, see :ref:`knowledge transfer`.
- The different trajectories files have been merged into one single file. There are no longer ``trajectories_with_gaps``, ``trajectories_wo_gaps`` nor ``trajectories_validated``, only ``trajectories``. This single file acts as ``trajectories_wo_gaps`` and it contains all modifications from the :ref:`validator_reference`.
- Since Protocol 3 does not exist anymore, the following parameters have been removed:

  - ``PROTOCOL3_ACTION``
  - ``THRESHOLD_ACCEPTABLE_ACCUMULATION``
  - ``MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS``
  - ``MAX_RATIO_OF_PRETRAINED_IMAGES``

- Remove the parameter ``ADD_TIME_COLUMN_TO_CSV`` setting it to always ``True``.
- Added property :attr:`~.Blob.identity_certainty` to :ref:`Blob` and used it to populate ``id_probabilities`` in :ref:`output structure`.
- Added automatic :ref:`Code reference` of main classes in documentation.
- Refactored and optimized :ref:`idmatcher.ai`.
- Added the new parameter ``TORCH_COMPILE`` to enable model compilation with ``torch.compile``, see :ref:`Advanced parameters`.
- Added the option to respect or ignore the tracking intervals while looking for errors in the :ref:`validator_reference`.
- Fixed GUI compatibility issues with PySide6.
- Works with PyQt6
- Allow files drag and drop into the GUIs to open videos, parameters files or sessions.
- Use ``gzip`` compression in trajectory HDF5 files.
- Rename time column in CSV trajectory file from ``seconds`` to ``time``.
- Rename ``MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION`` to ``MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION``.
- Added ``height``, ``width`` and ``silhouette_score`` to :ref:`trajectory files`.
- Fix Validator bug when acting on a centroid that has a duplicated identity.
- Fix and speed up blob-to-blob overlapping check.
- The default value for ``number_of_parallel_workers`` is limited to maximum 4 since the parallelized tasks involve disk read and write operations that are not CPU bound.
- Added automatic usage analytics, see :ref:`Usage Analytics`.
- Hardcoded ``LEARNING_RATE_IDCNN_ACCUMULATION`` to 0.005.
- Default Multiprocessing start method to ``forkserver`` when possible.
- Added a new frame preloader in the background of the Validator to improve video playback speed when browsing the list of errors.
- Improved GUI's Video players responsiveness, specially when dealing with heavy-loading video files.

The contrastive algorithm design for this version was developed by Jordi Torrents, Tiago Costa and Gonzalo G. de Polavieja. This work will be soon published.

5.2.12
======

- Works in Python 3.12, 3.11, and 3.10. And NumPy 2.0.
- Increase and decrease GUI font size with ``Ctrl++`` and ``Ctrl+-``.
- Fix https://gitlab.com/polavieja_lab/idtrackerai/-/issues/90
- Optimized `"Connecting coexisting fragments"`.
- The log file copy in session folder contains error tracebacks.
- Add ``DEVICE`` as an optional input parameter.
- Fix bug occurring when session folder contains non-ASCII characters.
- Add ``--no-labels`` parameter to :ref:`Video generator` (https://gitlab.com/polavieja_lab/idtrackerai/-/merge_requests/73 by https://gitlab.com/ssfrz)
- Another fix to "Too many open files" error by disabling ``pin_memory`` in Protocol 3 pre-training.
- Fix the automatic gap detector in Validator's interpolator when starting the interpolation by double clicking on a centroid.
- Fix individual videos cropping to keep the centroid of the animal in the center of the video when some padding has to be added.

5.2.11
======

- Fix critical bug making ``GUI_tools/icon.svg`` not be included in package build.
- Set ``pin_memory=False`` to non-training ``DataLoader`` to avoid "Too many open files" error, https://github.com/pytorch/pytorch/issues/91252.
- Limit the number of images per animal to use in :ref:`idmatcher.ai` to 10k.

5.2.10
======

- Added a :ref:`length calibration` tool in the :ref:`validator_reference` and its value ``length_unit`` in the trajectory files.
- Added the parameter ``bounding_box_images_in_ram`` to avoid saving bounding box images on disk.
- Added https://gitlab.com/polavieja_lab/midline to :ref:`data analysis`.
- Refactored tracking agent code and merged Protocol 1 into Protocol 2 (no effect on the algorithm).
- Cleaned ``Session.accumulation_folder`` attributes.
- Add color shuffling action in Validator.
- Validator interpolates now with splines using ``scipy.interpolate.make_interp_spline``. Previous backend ``scipy.interpolate.interp1d`` is now considered legacy by SciPy.

5.2.7
=====

- Improved branding design with a new logo and icon.
- Fix critical bug that was making knowledge transfer crash when tracking videos with different identification image sizes.
- Change default ``DATA_POLICY`` from ``"all"`` to ``"idmatcher.ai"``.
- Improved :ref:`validator_reference` stability.
- Reallocation of source code files.
- Catch crash in MacOS when ``BlockingIOError`` raises at opening H5FD files in mode `"r+"`.

5.2.6
=====

- Added resilience against corrupted videos avoiding wrongly decoded frames.
- Added advanced hyper-parameters ``THRESHOLD_EARLY_STOP_ACCUMULATION``, ``THRESHOLD_ACCEPTABLE_ACCUMULATION`` and ``MAXIMAL_IMAGES_PER_ANIMAL`` in docs and in terminal argument parser.
- More intelligent automatic zoom in Validator based on animal's body length.
- Cleaner logging by not printing the level name if it is ``DEBUG`` or ``INFO``.
- Allow float and infinite values in blob's area and intensity thresholds.
- Simplified area widget in Segmentation App.
- Define default session's name by double clicking in the session's name widget in the Segmentation App. The App will save `.toml` files without names if the widget is empty.
- ``THRESHOLD_EARLY_STOP_ACCUMULATION`` changed from 99.95% to 99.9%.
- Removed ``--settings`` argument from ``idtrackerai`` terminal command. Instead, multiple parameters files can be loaded with the ``--load`` argument in increasing order of priority.
- More compact representation of ``list_of_fragments.json``.
- Merged hyperparameters ``BATCH_SIZE_PREDICTIONS_IDCNN`` and ``BATCH_SIZE_PREDICTIONS_DCD`` into ``BATCH_SIZE_PREDICTIONS``
- Hardcoded ``LEARNING_RATE_IDCNN_ACCUMULATION`` to 0.005.

5.2.5
=====

- Simplified intensity thresholds in Segmentation App.
- Fixed identity removal in Validator.
- Deprecate ``list_of_blobs_validated.pickle``. Validated blobs are saved in the same original file ``list_of_blobs.pickle``
- Added "`Autoselect error`" option in the Validator.
- Fix crucial error of not properly shuffling images before splitting into train/validation datasets.

5.2.4
=====

- Improved memory efficiency.
- Removed ``IDCNN_NETWORK_NAME`` hyperparameter.
- Rename class ``Video`` to ``Session`` and ``video_object.json`` to ``session.json``.
- Rename parameter ``session`` to ``name``.
- Automatic session names made from video paths if no session name is provided.
- Do not override sessions if their name has been automatically set. Create consecutive names like `session_X_1`.
- Add a test step after each training.

5.2.2
=====

- Add ``--size`` parameter for individual :ref:`video generator`.
- The default value for ``number_of_parallel_workers`` is limited to 8 maximum.
- Do not save `list_of_blobs_no_gaps.pickle`.
- Merge network architectures for crossings and identification.
- Simplify video JSON file.
- Add boolean flag ``identity_transfer_succeded`` to video JSON file.
- Remove ``accumulation_statistics`` from video JSON file.
- Faster training with efficient image normalization.
- Faster coexisting Fragment connection.
- Faster `"First individual/crossing assignment"`.
- Do not fix identity of small Fragments.
- More memory efficient network predictions.

5.2.1
=====

- Fix crash in the video generator when using ``--gray``.


5.2.0
=====

- Add :ref:`exclusive regions of interest` new feature (https://gitlab.com/polavieja_lab/idtrackerai/-/merge_requests/58).
- Add a button to remove the selected centroid when double clicking (:ref:`validator_reference`).
- Allow loading a TOML parameters file with the `Open` button in the :ref:`segmentation app`.
- More informative logs
- Fix invalid model predictions when using Metal backend in MacOS machines https://gitlab.com/polavieja_lab/idtrackerai/-/issues/82.
- Fix Overflow crash when validation loss is exactly 0.
- Fix crash of Protocol 3 in Windows because of the default integer type in Numpy.
- Rename ``CustomError`` for ``IdtrackeraiError``.
- ``LEARNING_PERCENTAGE_DIFFERENCE_*`` hyperparameter renamed to ``LEARNING_RATIO_DIFFERENCE_*``

5.1.9
=====

- Allow idtrackerai to keep working even if OpenCV fails reading some video frames.
- Limit framerate option in GUI enabled by default.
- ``number_of_parallel_workers=1`` disables Python's Multiprocessing.
- Fix video generator when dealing with error frames.
- Add ``background_subtraction_stat`` to Segmentation App.
- More informative logs, specially in the accumulation results.
- Catch exception when it fails to read the number of frames of a video.
- Lighter ``ListOfBlobs`` and ``ListOfFragments`` files, cleaning ``cached_property`` before saving.

5.1.8
=====

- Fix NumPy integer types.
- Add defaults values in ``video_object.json``.
- Improve errors and tracebacks in log.

5.1.7
=====

- Works in Python 3.10 and 3.11.
- Improve error messages.
- New option to add a time column (in seconds) in the csv trajectory files. Parameter ``ADD_TIME_COLUMN_TO_CSV`` (``False`` by default).
- ``CONVERT_TRAJECTORIES_TO_CSV_AND_JSON`` default changed to ``True``.
- Reorganize trajectories output folder.
- Change video extension limitation for everything OpenCV can read.
- Fix ``output_dir`` error when it is stated in toml file.
- Fix Protocol 3 with knowledge transfer
- Merge and simplify ``learning_percentage_difference`` hyper-parameter
- More stable Validator with unfinished sessions.
- Clearer code in entry point functions and parameter management.

5.1.6
=====

- Fix Validator bugs
- Zoom un duplicates when clicking this error in Validator.
- Setup points as integers
- Fix input parameters effect on segmentation GUI
- Remove deprecated image blurring parameter

5.1.5
=====

- Fix ``Blob.is_an_individual`` setting when crossing detection training fails.
- Fix distorted image visualization in some video formats.
- Fix crash due to the missing ``Fragment.P1_vector`` attribute while validating.
- Abstract Qt dependencies with ``qtpy`` package.
- Default Qt package downgraded from PyQt6 to PyQt5 because its compatibility issues.
- Allow running non-GUI idtrackerai parts without any Qt installation.
- Add identity finder in Validator with `Ctrl+F`
- More versatile command line options for ``idtrackerai_csv``
- (testing) Allowing running idtrackerai in CPU only mode, AMD GPUs and MacOS with or without MPS acceleration.

5.1.4
=====

- ``ListOfGlobalFragments`` are saved in `.json` format.
- ``ListOfFragments`` are saved in `.json` format.
- Load the penultimate accumulation step if the last one broke.
- Focus on error when selecting an "No id" error in Validator.
- Reenable blobs' contours approximation using less points (lighter blobs objects in RAM and disk).
- Speed up trining by enabling ``persistent_workers=True`` in torch.utils.data.DataLoader.
- Improve Validator responsiveness when loading a large session.
- Remove scikit-learn dependency.
- Fix GUI initialization error in Fedora
- Fix background view in Segmentation App.
- Improve logging information.

5.1.3
=====

- Fix final compression bug on Windows.
- Fix ``idtrackerai_video`` incompatibility when tracking without identities.
- Fix GUI theme change malfunctioning.

5.1.2
=====

- Disables the first two changes of changelog :ref:`5.1.1` (the identification images construction method and the blob's contour approximation). These will be restored after some more testing.

5.1.1
=====

- Maximize the number of relevant pixels inside identification images (faster identification).
- Approximate blobs' contours using less points (lighter blobs objects in RAM and disk)
- Allow a variable number of animals (by setting n_animals=0) when tracking without identities.
- Removed anti-flickering filter. It improves intensity threshold's sensitivity and segmentation speed.
- Using *gzip* compression on identification images files when finishing a successful session. Optimizing loading times.
- Added playback speed in a action in the video player menu.
- Python's `datetime` usage in `Video` timers
- Free Enter/Return keys from GUI shortcuts.
- Fix log file issue when more than one session is running.
- Optimize rescaling when resolution reduction and adapt ROI to scale.
- Fix bug when validating single animal trackings.
- Fix "change font size" bug in GUIs.
- Ctrl+L to toggle playback framerate limitation (GUI).
- Fix cv2 BRG/RGB color confusion.
- Fix cv2 error in segmentation app when removing ROI while using background subtraction.

5.1.0
=====

- Implement idmatcher.ai
- ROIs can be reordered in segmentation app by drag and drop
- ROIs in segmentation app are ordered from bottom to top
- Select ROI by clicking inside the polygon in the video player (segmentation app)
- Fix typos
- Simplify idtrackerai/network file structure and imports
- Improve v4 compatibility reading video.json/npy
- Merged crossings/identification NetworkParams as a single dataclass
- Simplified GetPredictionIdentification converting it into a function
- Fix gray individual video generation

5.0.0
=====

- Full code revision promoting Python built-in libraries, argument type hints and multiples optimizations in terms of code simplicity and structure, RAM usage, lighter output generated data and faster execution.
- Unify all tools related to idtracker.ai in the same repository/package
- Works with Python 3.10
- New graphical apps buildings directly with PyQt6
- Remove dependency with

  - Pyforms
  - Python-video-annotator
  - Matplotlib
  - Joblib
  - Natsort
  - Tqdm
  - Pandas
  - Gdown

- Using the last versions of every remaining used dependency
- Completely new segmentation app, fully responsive, intuitive and faster
- Completely new validation app to view the session results, navigate through the possible tracking errors, fix them and manipulate the session using other extra tools.
- ``idtrackerai_csv`` tool to convert trajectories after the tracking process finished.
- Removed local_settings input method.
- New input methods (more direct and simple) (``--load``, ``--settings`` and terminal declarations).
- The tool "setup points" moved to the validator.
- No ``NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION``, background is computed sequentially.
- Merged ``NUMBER_OF_JOBS_FOR_SEGMENTATION`` and ``NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES`` in the same parameter ``NUMBER_OF_PARALLEL_WORKERS``.
- Easier background subtraction implementation, with "`median`" option. It is more robust against difficult tracking intervals/episodes/number of frames.
- Better and easier parallel `episode` definitions with optimized parallel distribution (specially with multiple files).
- Simplified attributes in all idtracker.ai objects.
- ListOfBlobs reconnects in almost no time after loading from saved *.pickle* file.
- Flexibility selecting the number of videos to track.
- Remove `Blob.pixels` attribute. Much faster and lighter blob manipulations.
- Stretch `Blob.bounding_box`. Much lighter segmentation images.
- Optimized 80% of the computational time of `_process_frame()` by properly removing the function `binary_fill_holes()`.
- Logs more readable, with more useful information and progress bars (using Rich).
- Faster h5py writing/reading implementation (by not opening and closing the h5py file for every single image, we keep them opened).
- Python objects are saved as pickle objects and json files when possible (lighter and more standard than .npy files).
- Removed option `save_areas`. Now, the statistics of the areas are always printed in the trajectory files.
- Parallel processing using built-in Multiprocessing, not Joblib.
- Reorganize internal modules promoting decoupling (fragmentation, tracking and postprocessing modules).
- Easy video generation with ``idtrackerai_video``.
- Package is defined using a *pyproject.toml* file.
- No git sub-modules used.
- Faster blob overlapping method (convexHull and point inside contour methods).

4.0.0
=====

- Works with Python 3.7.
- Remove Kivy submodules and stop support for old Kivy GUI.
- Neural network training is done with Pytorch 1.10.0.
- Identification images are saved as uint 8.
- Crossing detector images are the same as the identification images. This saves computing time and makes the process of generating the images faster.
- Improve data pipeline for the crossing detector.
- Parallel saving and loading of identification images (only for Linux)
- Simplify code for connecting blobs from frame to frame.
- Remove unnecessary execution of the blobs connection algorithm.
- Background subtraction considers the ROI
- Allows to save trajectories as csv with the advanced parameter `CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON` (using the `local_settings.py` file).
- Allows to change the output width (and height) of the individual-centered videos with the advanced parameter `INDIVIDUAL_VIDEO_WIDTH_HEIGHT` (using the `local_settings.py` file).
- Horizontal layout for graphical user interface (GUI). This layout can be deactivated using the `local_settings.py` setting  `NEW_GUI_LAYOUT=False`.
- Width and height of GUI can be changed using the `local_settings.py` using the `GUI_MINIMUM_HEIGHT` and `GUI_MINIMUM_WIDTH` variables.
- Add ground truth button to validation GUI.
- Added "Add setup points" featrue to store landmark points in the video frame that will be stored in the `trajectories.npy` and `trajectories_wo_gaps.npy` in the key `setup_poitns`. Users can use this points to perform behavioural analysis that requires landmarks of the experimental setup.
- Improved code formatting using the black formatter.
- Better factorization of the TrackerApi.
- Some bugs fixed.
- Better documentation of main idtracker.ai objects (`video`, `blob`, `list_of_blobs`, `fragment`, `list_of_fragments`, `global_fragment` and `list_of_global_fragments`).
- Dropped support for MacOS
