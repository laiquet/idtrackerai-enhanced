Running idTracker.ai
====================

The simple way
----------------
In the Conda environment where idtracker.ai is installed, running the command

.. code::

    idtrackerai

will execute the segmentation app, an app designed to help you define the correct input parameters for your video (more information about the app in :doc:`/segmentation_app`). From the app you can generate a parameters file to use it in *The advanced way* or you can directly run idtracker.ai.

The advanced way
----------------

While running the command :code:`idtrackerai`, users can define more advanced parameters in a more complex structure. Users can input parameters in these steps by increasing priority:

1. Loading a *.toml* settings file with the command argument :code:`--settings path/to/settings.toml`. Intended for some basic and constant settings that the user want to reuse in every tracking session (number_of_jobs_for_segmentation, background_stat, ...).
2. Loading a *.toml* parameter file with the command argument :code:`--load path/to/segmentation_app_file.toml` (possibly the output of the segmentation app). Intended for specific session dependent parameters (video_paths, intensity_ths, ...).
3. Specifying parameters via terminal. The user can be override any parameter in the command line. For example :code:`--video_paths videoA.avi --tracking_intervals [150,2000]`.
4. Any change

An advanced idtracker.ai usage example could be:

.. code-block:: bash
    :caption: advanced idtracker.ai example

    idtrackerai --settings my_basic_settings.toml --load video_A_parameters.toml --track_wo_identities true --number_of_animals 8 --track

Where the :code:`--track` flag indicates to track directly without launching the segmentation app.

LOCAL SETTINGS

idtracker.ai parameters
-----------------------

