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
    "sphinx_favicon",
]


project = pyproject["project"]["name"]
templates_path = ["_templates"]
nbsphinx_execute = "never"
source_suffix = ".rst"
copyright = "2018, Champalimaud Center for the Unknown"
author = "Jordi Torrents"
release = pyproject["project"]["version"]
version = pyproject["project"]["version"]
language = "en"
exclude_patterns = ["_build"]
pygments_style = "sphinx"
todo_include_todos = False
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "alt_text": "idtracker.ai - Home",
        "image_light": "_static/logo_light.svg",
        "image_dark": "_static/logo_dark.svg",
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
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Data repository",
            "url": "https://drive.google.com/drive/folders/1kAB2CDMmgoMtgFQ_q1e8Y4jhIdbxKhUv",
            "icon": "fa-solid fa-file-video",
        },
    ],
    "footer_start": ["copyright", "last-updated.html"],
    "footer_center": ["version"],
    "footer_end": ["sphinx-version", "theme-version"],
    "external_links": [{"name": "Polavieja Lab", "url": "https://polaviejalab.org/"}],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["theme-switcher", "search-button-field"],
}

favicons = [
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "favicon/apple-touch-icon.png",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "16x16",
        "href": "favicon/favicon-16x16.png",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "32x32",
        "href": "favicon/favicon-32x32.png",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "192x192",
        "href": "favicon/android-chrome-192x192.png",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "256x256",
        "href": "favicon/android-chrome-256x256.png",
    },
    {"rel": "mask-icon", "href": "favicon/safari-pinned-tab.svg", "color": "#5bbad5"},
    {"name": "msapplication-TileColor", "content": "#da532c"},
    # {"name": "theme-color", "content": "#ffffff"},
]


html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"
html_css_files = ["mycss.css"]


def external_role(name, rawtext, text: str, *args, **kargs):
    "Add a custom class to the link https://github.com/pydata/pydata-sphinx-theme/issues/1288"
    text = text.strip()
    content, link = text.split(" <")
    link = link[:-1]  # remove '>'

    node = raw(
        "",
        f'<a class="my-external-link" href="{link}" target="_blank">{content}</a>',
        format="html",
    )
    return [node], []


def setup(app):
    app.add_role("external", external_role)
