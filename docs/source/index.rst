:sd_hide_title:
:html_theme.sidebar_secondary.remove: true

************
idtracker.ai
************

.. This is for the website preview
.. raw:: html

    <img src="_static/favicon/wide_favicon_base.png" hidden>

.. div:: sd-text-center sd-text-primary sd-fs-4 sd-border-1 sd-rounded-3 sd-shadow-sm sd-px-3

    .. raw:: html

        <span style="font-family:nakula;font-size-adjust: 0.6;letter-spacing: 1.5px">idtracker.ai</span> tracks up to 100 unmarked animals from videos recorded in laboratory conditions using artificial intelligence. Free and open source.

    .. grid:: 2 4 4 4
        :gutter: 1 2 2 2
        :padding: 4 0 2 2

        .. grid-item::

            .. image:: _static/demo_gifs/zebra.webp
                :alt: Exmple of zebrafish tracked with idtracker.ai
                :target: https://youtu.be/Imz3xvPsaEw

        .. grid-item::

            .. image:: _static/demo_gifs/fly.webp
                :alt: Exmple of flies tracked with idtracker.ai
                :target: https://youtu.be/_M9xl4jBzVQ

        .. grid-item::

            .. image:: _static/demo_gifs/ants.webp
                :alt: Exmple of ants tracked with idtracker.ai
                :target: https://youtu.be/d0TTdu41NoA

        .. grid-item::

            .. image:: _static/demo_gifs/mice.webp
                :alt: Exmple of mice tracked with idtracker.ai
                :target: https://youtu.be/ANsThSPgBFM

.. grid:: 1 2 4 4
    :margin: 4 4 0 0
    :padding: 1
    :gutter: 3

    .. grid-item-card:: :fa:`download` Installation
        :link: install/installation
        :link-type: doc
        :text-align: center

        Install idtracker.ai and all its extra tools for Windows, MacOS and Linux.

    .. grid-item-card:: :fa:`book` User guide
        :link: user_guide/index
        :link-type: doc
        :text-align: center

        Learn how to use idtracker.ai, from beginner to advanced usages.

    .. grid-item-card:: :fa:`wrench` Included tools
        :link: user_guide/tools
        :link-type: doc
        :text-align: center

        Validate, generate videos, match identities across videos and analyze trajectories.

    .. grid-item-card:: :fa:`video` Good videos
        :link: good_videos/index
        :link-type: doc
        :text-align: center

        How to record good videos, experimental setups and example videos.

=========

.. grid:: 1 1 2 2

    .. grid-item::
        :child-align: end

        .. image:: _static/nature_logo_dark.svg
            :class: only-dark
            :target: https://doi.org/10.1038/s41592-018-0295-5
            :alt: Idtracker.ai publication in Nature Methods
            :width: 80%
            :align: center

        .. image:: _static/nature_logo_light.svg
            :class: only-light
            :target: https://doi.org/10.1038/s41592-018-0295-5
            :alt: Idtracker.ai publication in Nature Methods
            :width: 80%
            :align: center


        .. centered:: :external:`Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., idtracker.ai: tracking all individuals in small or large collectives of unmarked animals. Nature Methods 16, 179 (2019) <https://doi.org/10.1038/s41592-018-0295-5>` [:external:`PDF <https://drive.google.com/file/d/1DIHlykqhr9pVlxhMtY0R_G37JKno8vtX/view>`, :external:`arXiv <https://arxiv.org/abs/1803.04351>`]


    .. grid-item::
        ..

            The data used in this article can be found in this repository together with videos, their optimal tracking parameters and the resulting tracked sessions.

            .. button-link:: https://drive.google.com/drive/folders/1kAB2CDMmgoMtgFQ_q1e8Y4jhIdbxKhUv
                :color: primary
                :shadow:
                :expand:
                :click-parent:

                :fa:`file-video` Google Drive data repository

=======

.. grid:: 1 2 2 2
    :margin: 4 4 0 0
    :gutter: 5

    .. grid-item-card:: :fa:`code` Source Code
        :link: https://gitlab.com/polavieja_lab/idtrackerai
        :text-align: center

        The code is open and accessible at https://gitlab.com/polavieja_lab/idtrackerai, feel free to contribute.


    .. grid-item-card:: :fa:`laptop-code` Can animals really be recognized?
        :link: good_videos/identifying_fish
        :link-type: doc
        :text-align: center

        A Jupyter Notebook that proves that zebrafish can be distinguished using a CNN.

    .. grid-item-card:: :fa:`users` Google Groups
        :link: https://groups.google.com/g/idtrackerai_users
        :text-align: center

        Join the idtracker.ai users group to ask questions and get announcements about new releases.

    .. grid-item-card:: :fa:`people-group` Polavieja Lab
        :link: https://polaviejalab.org/
        :text-align: center

        This work belongs to Polavieja lab, Mathematics of Behavior and Intelligence (Champalimaud Foundation, Lisbon, Portugal).

    .. grid-item-card:: :fa:`envelope` Contact
        :link: mailto:idtrackerai@gmail.com
        :text-align: center

        If you encounter any problem or doubt, contact us at idtrackerai@gmail.com.


    .. grid-item-card:: :fa:`question` FAQs
        :link: user_guide/FAQs
        :link-type: doc
        :text-align: center

        See the frequently asked questions page to see if your issue has been already answered.

.. centered::
    Funded by Fundação Champalimaud and FCT under project PTDC/BIA-COM/5770/2020

.. toctree::
    :hidden:

    install/installation
    user_guide/index
    good_videos/index
