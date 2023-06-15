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

Yes. We are running idtracker.ai with all its features in gaming laptops from. Just read the :ref:`requirements` page.

Can I use idtracker.ai if my computer does not have a good GPU?
---------------------------------------------------------------

Yes, you can still use idtracker.ai if you don't have a GPU, see :ref:`install without a graphics device`.

Can idtracker.ai track multiple videos in batch?
------------------------------------------------

Yes, you can run idtracker.ai without any graphical interface so scripts can be built, check :ref:`usage`.

Does idtracker.ai give orientation and posture information?
-----------------------------------------------------------

Orientation and posture can be computed a posteriori from the *blobs_collection.npy* file
that idtracker.ai generates. In https://gitlab.com/polavieja_lab/midline.
we provide an example where we compute the nose, tail and midline for fish.

You can also generate a small video for every animal in :ref:`video generators` and use it to get the posture with one of the AI based posture trackings (:external:`Deeplabcut <http://www.mackenziemathislab.org/deeplabcut/>`, :external:`SLEAP <https://sleap.ai/>`, ...).

Does idtracker.ai track single animals?
---------------------------------------

Yes. Although idtracker.ai is designed to track multiple animals keeping their identities along the video, you can also track videos with a single animal. Just indicate that the number of animals to track is 1 in the corresponding text box in the :ref:`segmentation app`. The system will automatically skip the GPU intensive parts that are not necessary in this case. This means that you can use idtracker.ai to track single animals in a desktop or a laptop computer even if it does not have a GPU. In this case, idtracker.ai will run faster as it will only need to segment the video to extract the position of the animal.

Can I track humans with idtracker.ai?
-------------------------------------

We haven't tried to track people with idtracker.ai. We think that idtracker.ai can track well people in videos recorded under laboratory conditions. Tracking humans on natural environments (streets, parks,...) it is a much more difficult task for which idtracker.ai was not designed. However, as idtracker.ai is free and open source, you can maybe use parts of our algorithm to set your human tracking for natural environments.

Common installation problems
----------------------------

Some of the errors that you might encounter might have been already reported by other users and fixed. Please update your idtracker.ai to make sure you are using the latest version. To update idtracker.ai follow the instructions at the end of the :ref:`installation` page.

If the error persists, please report the issue in the repository https://gitlab.com/polavieja_lab/idtrackerai or send us an email to idtrackerai@gmail.com. We will try to fix it as soon as possible.
