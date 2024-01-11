
Trajectorytools
===============

While idtracker.ai's job ends when the trajectory files are validated. There is an extra tool (not part of but compatible with idtracker.ai) called *trajectorytools*.

It is a Python package to perform basic trajectory analysis and it is available at https://github.com/fjhheras/trajectorytools.

You can find some analysis routines from [1]_ implemented with *trajectorytools* in the Jupyter Notebook *trajectories.ipynb* in https://gitlab.com/polavieja_lab/idtrackerai_notebooks. Here we present some of the analysis we get using a 10 juvenile fish video:

.. image:: ../_static/ipynb/trajectories.png
    :height: 300
    :align: left

.. image:: ../_static/ipynb/density_of_neighbours.png
    :height: 300
    :align: right

.. div:: sd-text-center

    Smoothed trajectories (left) and density of neighbors around a focal fish (right)

.. figure:: ../_static/ipynb/velocity_and_acceleration.png
    :align: center
    :width: 80%

    Velocities and accelerations


.. figure:: ../_static/ipynb/polar_plots.png

    Polar distributions of positions, turnings and accelerations

.. figure:: ../_static/ipynb/distances_vs_random.png

    Inter-individual distance histograms compared with shuffled trajectories

.. rubric:: References

.. [1] Hinz, R. C., & de Polavieja, G. G. (2017). Ontogeny of collective behavior reveals a simple attraction rule. *Proceedings of the National Academy of Sciences*
