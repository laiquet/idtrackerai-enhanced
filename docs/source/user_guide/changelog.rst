*********
Changelog
*********

5.1.4
=====

- ``ListOfGlobalFragments`` are saved in `.json` format.
- Fix GUI initialization error in Fedora
- ``ListOfFragments`` are saved in `.json` format.
- Fix background view in Segmentation App.
- Return to the penultimate accumulation step if the last step as broken
- Focus on error when selecting an "No id" error in Validator

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

Author: Jordi Torrents

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

Author: Jordi Torrents

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

Author: Jordi Torrents

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
