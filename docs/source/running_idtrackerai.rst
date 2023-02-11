Running idtracker.ai
====================

The simple way
----------------
In the Conda environment where idtracker.ai is installed, running the command

.. code::

    idtrackerai

will execute the segmentation app, an app designed to help you define the correct input parameters for your video (more information about the app in :doc:`/segmentation_app`). From the app you can generate a parameters file to use it in *The advanced way* or you can directly run idtracker.ai.

The advanced way
----------------

Users can define more advanced (and optional) behavior and parameters using the command arguments

settings path_to_settings
    A *.toml* settings file can contain basic and constant settings intended to be used in every tracking session (number_of_jobs_for_segmentation, background_stat, ...).
load path_to_parameter_file
    Another *.toml* parameter file (which overrides the settings file) can contain specific parameters. Its main usage is for loading the parameters from the segmentation app. (video_paths, intensity_ths, ...)
PARAMETER VALUE
    Besides loading files, it is possible to declare any extra parameters in the command line.
track
    some text

.. --track as
..     some textdas

1. Specifying parameters via terminal. The user can be override any parameter in the command line. For example :code:`--video_paths videoA.avi --tracking_intervals [150,2000]`.
2. Any change

An advanced idtracker.ai usage example could be:

.. code-block:: bash
    :caption: advanced idtracker.ai example

    idtrackerai --settings my_basic_settings.toml --load video_A_parameters.toml --track_wo_identities true --number_of_animals 8 --track

Where the :code:`--track` flag indicates to track directly without launching the segmentation app.


LOCAL SETTINGS

idtracker.ai parameters
-----------------------

--load session_parameters
                    (path) Primary .toml file to load session parameters
--settings general_settings
                    (path) Secondary .toml file to load general settings
--track               Track the video without launching the GUI
--tracking_intervals 
                    (list_of_lists_of_two_ints) Tracking intervals in frames. Examples: '[0,100]', '[[0,100],[150,200],...]'. If none, the whole video is tracked
                    (default: None)
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