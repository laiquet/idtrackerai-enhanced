Video Generators
================

Idtracker.ai provides a tool to generate *tracked videos*. These are compositions of the original input video with the trajectories and labels on top. These videos can be useful to visually validate the tracking or to share your results. The command below:

.. code-block:: bash

    idtrackerai_video path/to/session_folder

generates a general tracked video inside the session folder (see the first Youtube video below).

.. warning:: General videos have a reduced quality

    General videos (those with the full view of the original video with the labels added) are generated with a reduced quality to reduce the file size. That makes them not suitable for further video postprocessing nor pose estimation. Use the individual videos for that purpose.

.. admonition:: Pose estimation
    :class: sidebar tip

    Individual videos can be useful to extract animal pose information with softwares like :external:`SLEAP <https://sleap.ai/>` and :external:`DeepLabCut <http://www.mackenziemathislab.org/deeplabcut/>`.


To generate individual videos, add the ``--individual`` flag to the command above. This creates a small individual video for each animal (see the second Youtube video below) and a Collage video (see the third Youtube video below).

To draw the original video in grayscale for both general and individual video styles, add the ``--gray`` flag to the command.

Optionally, you can specify the video time interval using the ``--s`` (start) and ``--e`` (end) flags. For example

.. code-block:: bash

    idtrackerai_video path/to/session_folder --individual --gray --s 0 --e 1000

generates individual videos in grayscale from frame 0 to frame 1000 with the trajectories from ``session_folder``.

.. raw:: html

    <div align="center">
    <iframe width="240" height="200" src="https://www.youtube.com/embed/U9JPLU5rlq0?loop=1&modestbranding=1&rel=0&playlist=U9JPLU5rlq0" frameborder="0" allowfullscreen></iframe>
    <iframe width="150" height="200" src="https://www.youtube.com/embed/yNRAtI5ZJh8?loop=1&modestbranding=1&rel=0&playlist=yNRAtI5ZJh8" frameborder="0" allowfullscreen></iframe>
    <iframe width="320" height="200" src="https://www.youtube.com/embed/cCOfTAU_3JA?loop=1&modestbranding=1&rel=0&playlist=cCOfTAU_3JA" frameborder="0" allowfullscreen></iframe>
    </div>

========

Run ``idtrackerai_video -h`` to display a list of all available options:

            --individual  Generate individual video. Default is a general video.
            --gray         Draw the original video in grayscale.
            --t            ``path`` Path to the trajectory file, default is session_dir/trajectories/trajectories*.
            --tl           ``int`` Trail length, number of points used to draw the individual trajectories traces in general videos. Default is 20.
            --s            ``int`` Frame where to start the video.
            --e            ``int`` Frame where to end the video.
            --size         ``int`` Size of the squared individual videos. Defaults to the median body length of the animals.
            --no-labels    Show centroids in general video without labeled IDs.
