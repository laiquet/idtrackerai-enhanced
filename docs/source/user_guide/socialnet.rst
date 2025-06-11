SocialNet
=========

.. |colab-badge| raw:: html

    <a href="https://colab.research.google.com/drive/1M_YIEehVGisKKXoly8e4Sd2MSpGPVVA4" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
    </a>

.. currentmodule:: socialnet

.. admonition:: Source code
    :class: sidebar note

    Check the source code at https://gitlab.com/polavieja_lab/socialnet.

SocialNet, presented in [1]_, is a model of collective behavior that is both predictive and insightful thanks to an organization in two modules. The first module gives a map of how two individuals interact in the collective while the second module gives how each individual aggregates the information of neighbours.

The model is trained using a set of trajectory files, from idtracker.ai or any other compatible source.

.. figure:: https://gitlab.com/polavieja_lab/socialnet/-/raw/master/socialnet_architecture.png
    :alt: SocialNet diagram
    :align: center

    **The SocialNet architecture** (extracted from Figure 1 in :ref:`[1]<socialnet_ref>`). **(A)** Variables used to predict future turns. Asocial variables, those only involving the focal, in red. Social variables, those involving both the focal and a neighbour, in blue. **(B)** Pair-interaction subnetwork of SocialNet, receiving asocial variables :math:`\alpha` and social variables :math:`\sigma_i` from a single neighbour :math:`i`, and outputting a single scalar value. All pair-interaction networks share the same weights. **(C)** Aggregation subnetwork of SocialNet. Same structure as B, but the input is a restricted symmetric subset of the variables and the output is passed through an exponential function to make it positive. **(D)** General SocialNet architecture, showing how the inputs of the pair-interaction and aggregation subnetworks are integrated to produce a single logit :math:`z` for the focal fish turning right after 1 s.


.. image:: https://gitlab.com/polavieja_lab/socialnet/-/raw/master/examples/interaction_subnetwork.png
    :align: left
    :height: 270


.. image:: https://gitlab.com/polavieja_lab/socialnet/-/raw/master/examples/aggregation_subnetwork_vars_fv_nbv_nbx_nby.png
    :align: right
    :height: 270

**Left:** Example interaction map showing how SocialNet predicts the influence of a neighbor on the focal animal's probability of turning right after 1 second (obtained with :func:`plot.plot_interaction_subnetwork`).

**Right:** Example aggregation map showing how SocialNet weighs different neighbor variables when aggregating social information (obtained with :func:`plot.plot_aggregation_subnetwork`).

Install SocialNet
-----------------

.. warning::

    SocialNet is **not included** in idtracker.ai, so it needs to be installed separately. It depends on Tensorflow which requires a Python version between 3.9 and  3.12. If you recently created your Conda environment for idtracker.ai you probably built it with Python 3.13. You can check the Python version of your Conda environment with the command ``python --version``. If you have Python 3.13 or higher, you will need to create a new Conda environment with Python 3.12 to install SocialNet:

    .. code-block:: bash

        conda create -n socialnet python=3.12
        conda activate socialnet

    If the idtracker.ai environment has Python between 3.9-3.12 you can reuse it and install SocialNet directly in it.


Install SocialNet by first installing TensorFlow following the :external:`TensorFlow installation guide <https://www.tensorflow.org/install/pip>`. And finally install SocialNet from our repository using pip:

.. code-block:: bash

    pip install git+https://gitlab.com/polavieja_lab/socialnet

Basic usage of SocialNet
------------------------

SocialNet has a simple API that allows users to train (:func:`model_train`) and test (:func:`model_test`) the model using Python.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Training and test SocialNet
   :nosignatures:

    model_train
    model_test

After training, the model can be analyzed by using the following plotting functions:

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Plot SocialNet results
   :nosignatures:

    plot.plot_aggregation_subnetwork
    plot.plot_interaction_subnetwork
    plot.plot_interaction_scores
    plot.plot_product


Here is an example of how to use SocialNet to train and test a model with trajectory files. In this example, we download some sample trajectory files from :external:`Google Drive data repository <https://drive.google.com/drive/folders/1kAB2CDMmgoMtgFQ_q1e8Y4jhIdbxKhUv>`, train a model, and then test it.

.. code-block:: python
    :caption: Example of training, testing and plotting SocialNet |colab-badge|

    from pathlib import Path
    import gdown
    from socialnet import model_test, model_train
    from socialnet.plot import (
        plot_aggregation_subnetwork,
        plot_interaction_scores,
        plot_product,
        plot_interaction_subnetwork,
    )

    # data from https://drive.google.com/drive/folders/1VH97_bNFz09Ke_kBL1oV2HbTS25BwnKC

    gdown.download(
        id="1y1ZhNr3eWbhYwA_ZPfIs9UjsinszKCwp", output="zebrafish_60_1.npy", resume=True
    )
    gdown.download(
        id="1aJb2pgzJE8dhDkWVGKWdBgYgFRhpWkj7", output="zebrafish_60_2.npy", resume=True
    )
    gdown.download(
        id="1LaNIqFGD5N9SUIq0yfIGEZUnrzF3TgLy", output="zebrafish_60_3.npy", resume=True
    )

    results_dict = model_train(
        ["zebrafish_60_1.npy", "zebrafish_60_2.npy", "zebrafish_60_3.npy"],
        session_name="example",
    )

    expected_output_folder = Path.cwd() / "socialnet_session_example"


    test_results = model_test(
        trajectory_files=["zebrafish_60_1.npy", "zebrafish_60_2.npy", "zebrafish_60_3.npy"],
        model_folder=expected_output_folder,
    )

    # plot slices of the neighbor x and y coordinates for different values of the focal velocity and the neighbor velocity
    # fv = focal velocity
    # nba = neighbour acceleration
    # nbv = neighbour velocity
    # nbx = neighbour x position
    # nby = neighbour y position
    fig_vars = ("fv", "nbv", "nbx", "nby")  # row_var, col_var, x_var, y_var

    plot_aggregation_subnetwork(expected_output_folder, fig_vars=fig_vars)
    plot_interaction_subnetwork(expected_output_folder)
    plot_interaction_scores(expected_output_folder, fig_vars=fig_vars)
    plot_product(expected_output_folder, fig_vars=fig_vars)

.. dropdown:: SocialNet CLI

    SocialNet can also has a command line interface (CLI). This provides a simple way to interact with the SocialNet API without needing to write Python code.


    .. code-block:: bash
        :caption: Main CLI commands

        socialnet train --help
        socialnet test --help


    .. code-block:: bash
        :caption: Plotting CLI commands

        socialnet plot --help
        socialnet plot aggregation_subnetwork --help
        socialnet plot interaction_subnetwork --help
        socialnet plot interaction_scores --help
        socialnet plot product --help

.. rubric:: References

.. _socialnet_ref:

.. [1] :external:`Heras FJH, Romero-Ferrero F, Hinz RC, de Polavieja GG (2019) Deep attention networks reveal the rules of collective motion in zebrafish. PLOS Computational Biology 15(9): e1007354. <https://doi.org/10.1371/journal.pcbi.1007354>`
