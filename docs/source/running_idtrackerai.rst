Running idtracker.ai
====================

Usage
-----

In the Conda environment where idtracker.ai is installed, running the command

.. code::

    idtrackerai

will execute the segmentation app, an app designed to help you define the correct input parameters for your video (more information about the :doc:`/segmentation_app`). From there, you can directly run idtracker.ai or save the specified parameters in a *.toml* file and load them later with

.. code::

    idtrackerai --load parameters.toml

to recover the segmentation app as you left it or

.. code::

    idtrackerai --load parameters.toml --track

to start the tracking process without any graphical interface (useful to run idtracker in remote via *ssh*)

More advanced parameters can be defined to extend idtracker.ai's capabilities. These can be defined in a settings *.toml* file which can be loaded with the :code:`--settings` flag.

Finally, any parameter can be defined in the terminal command by :code:`-PARAMETER VALUE`.

An example of an advanced idtracker.ai command could be:

.. code-block:: bash
    
    idtrackerai --settings my_basic_settings.toml --load video_A_parameters.toml --track_wo_identities true --number_of_animals 8 --track

.. note:: 

    Any parameter (advanced or basic from the segmentation app) can be defined in any of the three input methods. Loaded using the :code:`--load` file, the :code:`--settings` file or by terminal.

    The :code:`--load` parameters override the :code:`--settings` ones and any terminal declaration overrides both file loading methods



Complete list of idtracker.ai parameters
-----------------------

--load session_parameters
                    (path) Primary .toml file to load session parameters
--settings general_settings
                    (path) Secondary .toml file to load general settings
--track               
                    Track the video without launching the GUI
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