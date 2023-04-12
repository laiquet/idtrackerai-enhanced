import os
import sys

import toml
from docutils.nodes import raw

sys.path.append(os.path.abspath("./_ext"))

pyproject = toml.load(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "pyproject.toml"
    )
)
version = pyproject["project"]["version"]
project = pyproject["project"]["name"]

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx_design",
    "argparsers",
    "nbsphinx",
    "sphinx_toolbox.collapse",
    "sphinx_toolbox.wikipedia",
    "sphinx_togglebutton",
]

nbsphinx_execute = "never"
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
        "image_light": "_static/idtracker_logo_light.png",
        "image_dark": "_static/idtracker_logo_dark.png",
    },
    "show_nav_level": 2,
    "show_prev_next": False,
    "secondary_sidebar_items": ["page-toc"],
    "header_links_before_dropdown": 40,
    "primary_sidebar_end": [],
    "icon_links": [
        {
            "name": "Email",
            "url": "mailto:idtrackerai@gmail.com",
            "icon": "fa-solid fa-envelope",
        },
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
    ],
    "footer_start": ["copyright", "last-updated.html"],
    "footer_end": ["sphinx-version", "theme-version"],
    "external_links": [{"name": "Polavieja Lab", "url": "https://polaviejalab.org/"}],
}

html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"
html_css_files = ["mycss.css"]


def external_role(name, rawtext, text: str, *args, **kargs):
    # Add a custom class to the link
    text = text.strip()
    content, link = text.split(" <")
    link = link[:-1]  # remove '>'

    node = raw(
        "", f'<a class="my-external-link" href="{link}">{content}</a>', format="html"
    )
    return [node], []


def setup(app):
    app.add_role("external", external_role)
