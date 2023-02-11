Running idTracker.ai
====================

The beginner way
----------------
In the Conda environment where idtracker.ai is installed, running the command

.. code::

    idtrackerai

will execute the segmentation app, an app designed to help you define the correct input parameters for your video (more information about the app in :doc:`/segmentation_app`). From the app you can generate a parameters file to use it in *The advanced way* or you can directly run idtracker.ai.

The advanced way
----------------

Advanced users can define more parameters than the ones the segmentation app provides. These parameters and the segmentation ones can be defined in four levels:

    1. Settings file
    2. Segmentation app file
    3. Terminal arguments
    4. Segmentation app

where each level overrides the previous ones.


Idtracker.ai can be run following two main strategies, running from the app and from the terminal.


.. code::

    idtrackerai --load parameters.toml --settings settings.toml 