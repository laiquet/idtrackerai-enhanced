
It is very important for idtracker.ai to know the number of animals 
to be tracked. Make sure that the value in the box **Number of animals**
is equal to the number of animals that appear in the video (8 in this case). 
For a good performance of the algorithm, there must be multiple parts in the
video where the number of blobs detected (marked in red in the preview window) 
is equal to the **Number of animals** indicated in this text box.

You can get more information about the number of blobs detected by checking 
the option **Segmented blobs info**. Toggling this box will show a graph like 
this one:

.. figure:: ./_static/quickstart/area_graph.png
   :scale: 100 %
   :align: center
   :alt: area graph

If you only see a white window, move to a different frame for the graph to 
update the graph.

The title of the graph indicates the the number of blobs detected, together 
with the area of the smallest blob. In the graph, each bar indicates the area 
in pixels of each of the detected blobs. The horizontal gray line indicates the 
minimum area.

Check the :doc:`./GUI_explained` section to get more information about the 
**Check segmentation** option.

There are four main parameters that affect the number of blobs detected in a 
given frame. The **Intensity thresholds** (minimum and maximum) and the 
**Area thresholds** (minimum and maximum). Connected pixels which intensity 
values are within the range defined by the intensity thresholds will be 
detected as a blob if the number of pixels that define the blob (the area of 
the blob) is within of the range defined by the area thresholds.

To modify the different thresholds, you can type the new value inside of the 
text box, scroll up/down with the cursor placed on top of the box, or drag 
the extremes of the blue bars.

Check the :doc:`./GUI_explained` section to get more information about the 
**Subtract background** box and the **Resolution reduction** parameter.

Sometimes you might want to discard the beginning or the end of a video. 
You can do this by setting the starting and ending frames of the 
**Tracking interval**.

Check the :doc:`./GUI_explained` section to get more information about the 
**Multiple** box that will allow you to set multiple tracking intervals.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 6. Set a region of interest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example video, the animals can be easily separated from the background 
using only the **Intensity thresholds** and the **Area thresholds**. However, 
it can happen that there are other detected blobs in the frame that do not 
correspond to any animal (e.g. reflections, parts of the experimental rig,...). 
If these objects appear consistently in a part of the frame where the animals 
do not appear, you can mask the objects by setting one or multiple regions of 
interest (ROI).

Toggle the box **Apply ROI**. Three buttons and a white box will appear below.

.. figure:: ./_static/quickstart/apply_roi.png
   :scale: 100 %
   :align: center
   :alt: apply roi

Click on the **Rectangle** button. Then, in the preview window, click on one 
of the corners of the rectangle that you want to draw and drag to the position 
of the opposite corner. This should draw a green rectangle.

.. figure:: ./_static/quickstart/roi.png
   :scale: 100 %
   :align: center
   :alt: roi

Only the pixels inside of the ROI will be considered when applying the 
**Intensity thresholds** and the **Area thresholds**. To delete the ROI, 
click on the list of points created in the white box. They will highlight 
in blue. Then click the minus sign (-) button on the top right of the box to 
delete it. If you do not want to apply any ROI, uncheck the **Apply ROI** box.

Check the :doc:`./GUI_explained` section to get more information about how to 
draw **Polygons** and **Ellipses**.

*NOTE: To track the example video with good performance results you don't need 
to set any ROI*

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 7. Set the session name and start tracking the video
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before pressing the **Track video** button, add the name of the tracking 
session in the top right *Session* text box. The results of the tracking will 
be saved in a folder with the name "Session_sessionname" where "sessionname" 
will be the text that appear in the *Session* text box.

idtracker.ai allows the user to save the preprocessing parameters as they 
appear in the main window. This can be done with the **Save parameters** 
button. Saving the preprocessing parameters is useful to track the video 
later from the command line. Check the :doc:`tracking_from_terminal.rst` 
section to get more information about how to save the parameters and track 
multiple videos sequentially.

For now, click the **Track video** button to start tracking the video. The 
system will compute the different steps necessary to track the video and the 
**Progress** bar will advance accordingly. Note that no feedback is given to 
the user in the form of windows or graphs. You can check the progress
of the tracking in the terminal.

In Linux you use the commands

.. code-block:: bash

    top

or

.. code-block:: bash

    htop

to monitor the CPU and memory usage. And the command

.. code-block:: bash

    watch -n -1 nvidia-smi

to monitor the GPU usage.

In Windows you can check Windows System Resource Manager.

At the end of the tracking, a window will pop up showing that the tracking 
has finished and the estimated accuracy. Also, the terminal will show a 
message indicating the estimated accuracy and the value of the DATA_POLICY 
advanced parameter (see :doc:`advanced_parameters`).

.. figure:: ./_static/quickstart/output_test.png
   :scale: 100 %
   :align: right
   :alt: finished terminal

Check the :doc:`./GUI_explained` section to get more information about the 
effects of toggling the box *Track without identities*.

Check the :doc:`./advanced_parameters` section to get more information about 
how to change some advanced parameters of the algorithm.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 8. Validate the trajectories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the tracking has finished, the button **Validate trajectories** will 
activate. This button will open a new window that will show the results of 
the tracking for every frame of the video. You will be able to correct the 
identities of the animals that were misidentified and to change the position 
of the centroids of individual and crossing animals.

Check the instructions of the validation GUI in :doc:`./validation_GUI.rst` 
page.

^^^^^^^^^^^^^^^^^^^^
Step 9. Output files
^^^^^^^^^^^^^^^^^^^^
The data generated during the tracking process and the trajectories files are 
stored in the session folder. If the name of the session was "quickstart" the 
name of the folder will be "Session_quickstart". Depending on the value of the 
DATA_POLICY advanced parameter (see :doc:`./advanced_parameters`), the content 
of the session folder will vary. In this case, the content of the folder 
should be similar to this one.

.. figure:: ./_static/quickstart/session_folder.png
   :scale: 100 %
   :align: center
   :alt: session folder

The trajectories are stored in the subfolders "trajectories" and 
"trajectories_wo_gaps". The "trajectories.npy" file contains the trajectories 
with gaps (NaN) when the animals were touching or crossing. 
The "trajectories_wo_gaps.npy" file contains the trajectories with the 
gaps interpolated. There might still be some gaps where the interpolation 
was not consistent.

Check the :doc:`trajectories_analysis` section to learn more about how to 
load and analyze the trajectories generated with idtracker.ai.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Try the 100 zebrafish sample video
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can download the video from 
`this other link <https://drive.google.com/open?id=1Tl64CHrQoc05PDElHvYGzjqtybQc4g37>`_. 
Note that the size of this video is 22.4GB, so it 
should take around 30 minutes to download it at an average rate of 
12Mb/s.

Due to the higher frame size of this video (3500x3584) you might notice a 
decrease of speed when adjusting the preprocessing parameters.

**Tracking time and preprocessing parameters...**
