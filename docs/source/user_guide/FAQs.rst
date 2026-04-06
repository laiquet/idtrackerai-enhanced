****
FAQs
****

Can I use idtracker.ai in my videos?
------------------------------------

You can check our :ref:`example videos` to see the type of videos in which idtracker.ai worked well. We also give a set of :ref:`guidelines for good videos` that we advise users to follow to get the best results with idtracker.ai.

Does idtracker.ai work in Windows?
----------------------------------

Yes, in the :ref:`installation` we provide instructions to install idtracker.ai in Windows. We have tested the installation in computers running Windows 10 and Windows 11.

Can I run idtracker.ai in a laptop?
-----------------------------------

Yes. We are running idtracker.ai with all its features on gaming laptops. Just read the :ref:`requirements` page.

Can I use idtracker.ai if my computer does not have a good GPU?
---------------------------------------------------------------

Yes, you can still use idtracker.ai if you don't have a GPU, see :ref:`install without a graphics device`.

Can idtracker.ai track multiple videos in batch?
------------------------------------------------

Yes, you can run idtracker.ai without any graphical interface so scripts can be built, check :ref:`usage`.

Does idtracker.ai give orientation and posture information?
-----------------------------------------------------------

Orientation and posture can be computed afterwards from the *list_of_blobs.pickle* file
that idtracker.ai generates. In https://gitlab.com/polavieja_lab/midline
we provide an example where we compute the nose, tail and midline for fish.

You can also generate a small video for every animal in :ref:`video generator`. Use it to get the posture with one of the AI based posture trackings (:external:`Deeplabcut <http://www.mackenziemathislab.org/deeplabcut/>`, :external:`SLEAP <https://sleap.ai/>`, ...).

Does idtracker.ai track single animals?
---------------------------------------

Yes. Although idtracker.ai is designed to track multiple animals keeping their identities along the video, you can also track videos with a single animal. Just indicate that the number of animals to track is 1 in the corresponding text box in the :ref:`segmentation app`. The system will automatically skip the GPU intensive parts that are not necessary in this case. This means that you can use idtracker.ai to track single animals in a desktop or a laptop computer even if it does not have a GPU. In this case, idtracker.ai will run faster as it will only need to segment the video to extract the position of the animal.

Can I track humans with idtracker.ai?
-------------------------------------

We haven't tried to track people with idtracker.ai. We think that idtracker.ai can track well people in videos recorded under laboratory conditions. Tracking humans in natural environments (streets, parks, etc.) is a much more difficult task for which idtracker.ai was not designed. However, as idtracker.ai is free and open source, you can maybe use parts of our algorithm to set your human tracking for natural environments.

Common installation problems
----------------------------

Some of the errors that you might encounter might have been already reported by other users and fixed. Please update your idtracker.ai to make sure you are using the latest version. To update idtracker.ai follow the instructions at the end of the :ref:`installation` page.

If the error persists, please report the issue in the repository https://gitlab.com/polavieja_lab/idtrackerai or send us an email to info@idtracker.ai. We will try to fix it as soon as possible.

Can I track multiple videos without using the graphical interface?
------------------------------------------------------------------

Yes. Save your session parameters as a TOML file from the :ref:`segmentation app` (using *Save parameters*), then run:

.. code-block:: bash

    idtrackerai --load parameters.toml --track

You can script this call in a loop or a shell script to process many videos in batch. See :ref:`terminal usage` for details.

How do I know if the tracking was successful?
---------------------------------------------

After tracking, idtracker.ai reports the global identification accuracy at the end of the session log. A successful session will produce trajectory files in ``session_<name>/trajectories/``. Open the session in the :ref:`validator_reference` to visually inspect errors — it will show a list of frames where the identity could not be determined with confidence.

Can I track videos that have multiple separate ROIs?
----------------------------------------------------

Yes. In the :ref:`segmentation app` you can define multiple regions of interest (ROIs). If the animals are confined to separate regions and should not interact, you can enable *exclusive ROIs* so that animals are tracked independently within each region.

How do I track videos that are split into multiple files?
---------------------------------------------------------

You can provide multiple video paths in the ``video_paths`` field of the TOML parameter file, or select multiple files in the :ref:`segmentation app`. idtracker.ai will treat them as a single continuous video.

How do I update idtracker.ai?
------------------------------

Follow the update instructions at the end of the :ref:`installation` page. In general, activate your conda environment and run:

.. code-block:: bash

    pip install --upgrade idtrackerai

Can idtracker.ai handle very long videos?
-----------------------------------------

Yes, the maximum video length is limited only by available RAM and disk space. For very long videos, consider using the ``tracking_intervals`` parameter to track specific time windows, or split the processing across multiple sessions.

How can I reduce memory usage during tracking?
-----------------------------------------------

See the ``number_of_parallel_workers`` parameter in :ref:`parallel processing`. Reducing the number of parallel workers lowers RAM usage during segmentation (each worker can use up to 4 GB). You can also set a stricter ``data_policy`` to avoid keeping large intermediate files on disk.
