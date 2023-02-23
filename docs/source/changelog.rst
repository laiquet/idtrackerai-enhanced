:orphan:

*********
Changelog
*********

v5.0.0
======

- Works with Python 3.10
- GUIs directly with PyQt6
- Neural network training is done with last version of Pytorch
- idtrackerai_csv
- no local_settings
- setup points in validator
- no NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION
- fused number_of_jobs_for_segmentation and NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES
- Easier to use background subtraction implementation, with "median" option. It is more robust against difficult tracking intervals/episodes/number of frames
- Better and easier `episode` definitions with optimized parallel distribution (specially with multiple files)
- Simplified `Video.video_paths` attribute removing the old attribute `Video.video_path`
- Simplified segmentation code using the new `episode` definition
- List of blobs can reconnect after loading from saved *list_of_blobs.pickle* in almost no time
- Flexibility when selection the number of videos to track
- Remove Blob.pixels. Much faster and lighter blob manipulations
- Stretch Blob.bounding_box. Much lighter segmentation images
- Optimized 80% of the computational time of `_process_frame()` by properly removing the function `binary_fill_holes()`
- Logs more readable, with more useful information and progress bars
- Faster h5py writing/reading implementation (by not opening and closing the h5py file for every single image, we keep them opened)
- Remove dependency with matplotlib
- Python objects are saved as pickle objects and json files when possible (lighter and more standard than .npy files)
- Improved trajectories video generators
- Automatic `save_areas` output management
- Parallel processing uses Multiprocessing, not Joblib

v4.0.0
======

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