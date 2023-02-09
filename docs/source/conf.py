import idtrackerai


version = idtrackerai.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "numpydoc",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
project = "idtrackerai"
copyright = "2018, Champalimaud Center for the Unknown"
author = "Francisco Romero Ferrero, Mattia G. Bergomi"
version = version
release = version
language = "en"
exclude_patterns = ["_build"]
pygments_style = "sphinx"
todo_include_todos = False
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {"image_light": "./_static/2fish.png", "image_dark": "./_static/2fish.png"},
    "secondary_sidebar_items": [],
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],  # "navbar-nav"
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
}
html_context = {"default_mode": "auto"}
html_sidebars = {"**": ["globaltoc.html", "sourcelink.html", "searchbox.html"]}
html_title = "%s v%s Manual" % (project, version)
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
