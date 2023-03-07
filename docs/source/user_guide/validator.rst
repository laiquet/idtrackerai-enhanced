    Under construction...

*********
Validator
*********

.. role:: toml(code)
   :language: toml


.. admonition:: Warning
    :class: sidebar warning

    This tool may be overwhelming for beginner users, there's not need to use is to get decent trajectories.


Idtracker.ai's validator is a graphical application to check, modify and validate a successful tracking session. It loads the ``list_of_blobs`` and the ``video_object`` from the session folder so it do **NOT** set :toml:`data_policy = 'trajectories'` if you want to use this tool.

To start the app, run the next command:

.. code-block:: bash

    idtrackerai_validator path/to/session_folder

to open the desired session, or just

.. code-block:: bash

    idtrackerai_validator

to open a blank validator and manually load a session with :kbd:`Ctrl+O`.

.. figure:: ../_static/validator_dark.png
    :class: only-dark

.. figure:: ../_static/validator_light.png
    :class: only-light

    idtracker.ai's validator application


.. grid::
    :gutter: 0
    :class-container: sd-text-center
    :outline:
    :padding: 0

    .. grid-item::
        :padding: 2
        :columns: 12

        .. button-ref:: app_actions_link
            :color: primary
            :outline:

        Actions related to the application and its operation

    .. grid-item::
        :columns: 4

        .. grid::
            :margin: 0
            :gutter: 0
            :outline:

            .. grid-item::
                :padding: 2

                .. button-ref:: list_of_errors_link
                    :color: primary
                    :outline:
                
                An error analyzer and explorer

        .. grid::
            :padding: 0
            :gutter: 0

            .. grid-item::
                :padding: 2
                :columns: 12

                .. button-ref:: interpolator_link
                    :color: primary
                    :outline:
                
                The interpolation tool to close *NaN* gaps



    .. grid-item::
        :padding: 2
        :outline:
        :columns: 4

        .. button-ref:: video_player_link
            :color: primary
            :outline:

        The interactive video player displaying the current video frame with all extra information on top

    .. grid-item::
        :columns: 4

        .. grid::
            :margin: 0
            :gutter: 0
            :outline:

            .. grid-item::
                :padding: 2
                :columns: 12

                .. button-ref:: extra_tools_link
                    :color: primary
                    :outline:
                
                A collection of three minor impact tools

        .. grid::
            :gutter: 0

            .. grid-item::
                :padding: 2
                :columns: 12

                .. button-ref:: blob_extra_info_link
                    :color: primary
                    :outline:
                
                Displayer of selected blob's main attributes



.. _app_actions_link:

App actions
===========

Here you'll find the application options. None has an effect on the data being validated and most of them have an associated shortcut.

- **About**: link to this webpage and update checker.
- **View**: access to quit the app, change the font size and toggle the dark theme.
- **Video Player**

  - **Enable Color**: toggles color in the video player.
  - **Limit framerate**: limits the frame rate to the original framerate of the input video (default to ``True`` because the there's minimum processing while playing the video and maximum framerate could be too much).
  - **Reduce memory usage**: A cache system is implemented in the video player to access the previously displayed frames faster. The size of this cache is limited to the last 128 frames. Enable this options to reduce this to the last 16 frames.

- **Session**: open a session by browsing the desired session folder and save the current session as well as generating the corresponding validated trajectory file.
- **Draw**: toggle different blob's attributes to draw in the video player. Regions of interest can also be drawn when present.

.. _list_of_errors_link:

List of errors
==============

A list of all errors in the current session classified in four error types:

- ``No id`` A blob's centroid could not be identified or has a invalid identity.
- ``Miss id`` The animal with identity ``Id`` couldn't be located (*NaN* gap).
- ``Jump`` The speed of animal the animal with identity ``Id`` is suspiciously large.
- ``Dupl`` (Duplicated) There are more than one centroid identified as animal ``Id``.

``Jump`` type errors are triggered when an animal moves faster than the mean value of all speed values in the session plus :math:`x` times the standard deviation. This threshold :math:`x` can be modified by the user in the **Jumps threshold** slider.

Clicking an error will make video player focus on it and, if a type ``Miss id`` or ``Jump`` is clicked, the :ref:`interpolator` will be activated. As some ``Jumps`` errors are not real errors, already user interpolated jumps (even if they still are over the threshold) will no longer appear as errors. User can reset the set of user accepted jumps by clicking :kbd:`Reset`.

.. _interpolator_link:

Interpolator
============

The interpolator can correct trajectories and close *NaN* gaps using polynomial interpolations. It can be activated by clicking an errors of type ``Miss id`` or ``Jump`` in the :ref:`list of errors` or by double clicking a centroid on the :ref:`video player` and clicking `"Interpolate here"`.

When activated, the interpolator will focus on a single animal identity and will take some input data from the current animal trajectory (drawn as red dots in the video player) and will propose the position of the missing centroids inside the interpolation range (drawn as white dots).

User can modify the interpolation parameters (*"Interpolation order"* and *"Input size"*). Also, user can adjust the trajectories manually by removing the current centroid (move through the video with :kbd:`A` and :kbd:`D` to select the centroid you want to remove) and by setting the current centroid position by clicking in the video player (only inside the interpolation range). Click *"Apply"* to accept the interpolation proposal and click another errors to continue validating.

.. _video_player_link:

Video player
============

In the video player the video frames will be live displayed as well as the blobs information (contours, labels...). Double clicking on one centroid will display the options to modify it. User can change the identity of the centroid (all propagate this change up to next crossing, its expands on the entire fragment) and also the :ref:`Interpolator` can be called from here.

.. _extra_tools_link:

Extra tools
===========

The next tools have no effect on the trajectories nor on any other aspect of the session. Their information will be included in the :ref:`trajectory files` for user to use it as desired.

Groups
------

Create identity groups by clicking :kbd:`Add`, writing the group name, and clicking on every identity so select/deselect it. When done, uncheck the :kbd:`Edit` button to finish editing the group.

Labels
------

Set a label (a name) for every identity in your session.

Setup points
------------

Create a set of *"Setup points"* by clicking :kbd:`Add`, writing the desired name and clicking on the video player to set the desired positions of the points. This could be used to mark the cornars/center of your experimental arena, some obstacle, or the position of some rule to calibrate distances.

.. _blob_extra_info_link:

Blob's extra info
=================

When clicking a centroid on the video player, this tool will display its main attributes (mostly for debugging purposes). The selected identity will be followed through the video displaying information of the current blob with the selected identity. 

Validator shortcuts
===================

.. list-table:: 
    :widths: auto
    :header-rows: 1

    * - Key
      - Action
    * - :kbd:`Q`
      - Quit the app
    * - :kbd:`Ctrl+O`
      - Open session
    * - :kbd:`Ctrl+S`
      - Save trajectories
    * - :kbd:`Alt+L`
      - Toggle labels drawing
    * - :kbd:`Alt+C`
      - Toggle contours drawing
    * - :kbd:`Alt+P`
      - Toggle centroids drawing
    * - :kbd:`Alt+B`
      - Toggle bounding boxes drawing
    * - :kbd:`Alt+T`
      - Toggle trails drawing
    * - :kbd:`Alt+R`
      - Toggle ROIs drawing
    * - :kbd:`Space`
      - Play/pause video player
    * - :kbd:`1` - :kbd:`9`
      - Change the video playback speed
    * - :kbd:`Right` / :kbd:`D`
      - Move video playback forward
    * - :kbd:`Left` / :kbd:`A`
      - Move video playback backward
    * - :kbd:`U`
      - Update list of errors
    * - :kbd:`Enter`
      - Apply interpolation (when interpolating)
    * - :kbd:`Esc`
      - Abort interpolation (when interpolating)
    * - :kbd:`R`
      - Remove current centroid (when interpolating)
