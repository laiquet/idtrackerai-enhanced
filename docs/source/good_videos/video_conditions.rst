**************************
Guidelines for good videos
**************************

In the idtracker.ai paper we checked that the algorithm can perform well even if the videos are compressed (Supplementary Table 10), low resolution (Supplementary Figure 7), slightly out of focus or blurred (Supplementary Figure 8), or the illumination conditions are not completely homogeneous (Supplementary Figure 9).

Still, we recommend following some guidelines during the recording of videos to maximize the probability of success and the accuracy of the tracking.

Resolution
==========

The higher the number of pixels per individual, the more information to distinguish it from the rest. Notice that, on the downside, the additional information makes the algorithm less time-efficient. We recommend recording with the highest resolution possible and downscaling it later using the :ref:`resolution reduction` parameter if needed.

Frame rate
==========

The frame rate must be high enough for the blobs associated with the same individual to overlap in consecutive frames, when moving at average speed. A low frame rate with respect to the average speed of the animals can cause a bad fragmentation of the video: An essential process in the tracking pipeline, that allows collecting images belonging to the same individual and organizing them into fragments. On the contrary, excessively high frame rates will make the information coming from the analysis of the fragments highly redundant. This will increase the computational time necessary to track the video, without guaranteeing an improvement of the identification of the individuals. In the examples provided in this paper, the frame rate ranges between 25 to 50 :abbr:`fps (frames per second)`.

Duration
========

The maximum duration of the videos for which idtracker.ai works will only depend on the RAM and disk memory available. The longer the videos, the more information idtracker.ai will have to learn and to identify meaning better results. On the other side, short videos can be tracked as far as the algorithm can gather enough images to train the identification network. idtracker.ai tracks 20 second videos of 8 zebrafish at 32 fps with no problem.

Video format
============

The system works with any video format compatible with last version of OpenCV (*.avi*, *.mp4*, *.mpg*, *.mov*). We recommend uncompressed or lossless video formats: Some compression algorithms work by deleting pieces of information that could be crucial for the identification of the individuals. However, we have successfully tracked videos with compressed formats: .avi (FPM4 video codec) and .MOV (avc1 video codec).

Illumination
============

Illumination has to be as uniform as possible, so that the appearance of the animals is homogeneous along the video. We recommend using indirect light either by making the light reflect on the walls of the setup, or by covering the setup with a light diffuser. Although we have also tracked videos with retro-illuminated arenas, recall that the tracking systems relies on visual features of the animals that this type of illumination could hide.

Definition and focus
====================

Images of individuals should be as sharp and focused as possible for their features to be clearly displayed along the entire video. When using wide apertures on the camera, the depth of field can be quite narrow. Make sure that the plane of the sensor of the camera is parallel to the plane of the arena so that animals are focused in all parts of it. In addition, exposition time (shutter speed) should be high enough so that animals do not appear blurred when moving at average speed. Blurred and out of focus images are more difficult to be identified correctly.

Background
==========

The background should be as uniform as possible. To facilitate the detection of the animals during the segmentation process, the background color has to be chosen in order to maximize the contrast with the animals. Small background inhomogeneity or noise are acceptable and can be removed by the user during the selection of the tracking parameters in the :ref:`segmentation app`.

* Static or moving objects much smaller or much larger than the animals can be removed by setting the appropriate maximum and minimum blob :ref:`area thresholds`.

* Static objects of the same size and intensity of the animals can be removed by using the background subtraction in :ref:`background subtraction and intensity thresholds`.

* Regions of the frame can be also excluded by selecting a :ref:`regions of interest`.

Shadows
=======

Shadows projected by the individuals on the background can lead to a bad segmentation and, hence, to a bad identification. Shadows can be diffused by using a transparent base separated from an opaque background or by using a retro-illuminated arena.

Reflections
===========
Reflections of individuals on the walls of the arena should be avoided: They could be mistaken for an actual individual during the segmentation process. Reflections in opaque walls can be reduced by using either very diffused light or matte walls. For aquatic arenas with transparent walls, reflections can be softened by having water at both sides of the walls. Furthermore, reflections can be removed by selecting a detailed :ref:`regions of interest`.

Variability in number of pixels per animal
==========================================

The number of pixels in a blob is one of the criteria used to distinguish individual animals from crossings. An optimal video should fulfill the two following conditions. First, the number of pixels associated with each individual should vary as little as possible along the video. Second, the size of an individual should vary as little as possible depending on its position in the arena. In any case, strategies to avoid misidentification are put in place, even in case of variable animal sizes.
