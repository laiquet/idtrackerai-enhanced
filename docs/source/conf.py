import os

import toml

pyproject = toml.load(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "pyproject.toml"
    )
)
version = pyproject["project"]["version"]
project = pyproject["project"]["name"]


extensions = [
    # "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    # "numpydoc",
]
source_suffix = ".rst"
master_doc = "index"
copyright = "2018, Champalimaud Center for the Unknown"
author = "Francisco Romero Ferrero, Mattia G. Bergomi"
release = version
language = "en"
exclude_patterns = ["_build"]
pygments_style = "sphinx"
todo_include_todos = False
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "image_light": "2fish.png",
        "image_dark": "2fish.png",
        "text": "idTracker.ai",
    },
    "secondary_sidebar_items": [],
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],  # "navbar-nav"
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "icon_links": [
        {
            "name": "Google Groups",
            "url": "https://groups.google.com/g/idtrackerai_users",
            "icon": "fa-solid fa-users",
        },
        {
            "name": "GitLab",
            "url": "https://gitlab.com/polavieja_lab/idtrackerai",
            "icon": "fa-brands fa-gitlab",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/idtrackerai/",
            "icon": "fa-solid fa-box",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/idtrackerai",
            "icon": "fa-brands fa-twitter",
        },
        {
            "name": "Youtube",
            "url": "https://www.youtube.com/@idtrackerai5235",
            "icon": "fa-brands fa-youtube",
        },
    ],
}
html_context = {"default_mode": "auto"}
html_sidebars = {"**": ["globaltoc.html", "sourcelink.html", "searchbox.html"]}
# html_title = "%s v%s Manual" % (project, version)
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "idtrackerai"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
latex_documents = [
    (
        master_doc,
        "idtrackerai.tex",
        "idtrackerai Documentation",
        " Francisco Romero-Ferrero, Mattia G. Bergomi",
        "manual",
    )
]
man_pages = [(master_doc, "idtrackerai", "idtrackerai Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "idtrackerai",
        "idtrackerai Documentation",
        author,
        "idtrackerai",
        "One line description of project.",
        "Miscellaneous",
    )
]

# google analytics
googleanalytics_id = "UA-114600635-1"
# autoclass_content = "both"

# def skip(app, what, name, obj, would_skip, options):
#     if name == "__init__":
#         return False
#     return would_skip

# def setup(app):
#     app.connect("autodoc-skip-member", skip)

# autodoc_default_options = {
#     'members': True,
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': False,
# }
