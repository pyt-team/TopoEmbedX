"""Sphinx configuration file."""

# -- Project information -----------------------------------------------------

project = "TopoEmbedX"
copyright = "2022-2023, PyT-Team, Inc."
author = "PyT-Team Authors"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "toponetx": ("https://pyt-team.github.io/toponetx/", None),
}

# Configure nbsphinx for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_baseurl = "https://pyt-team.github.io/topoembedx/"

html_context = {
    "github_user": "pyt-team",
    "github_repo": "TopoEmbedX",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pyt-team/TopoEmbedX",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": True,
}

html_show_sourcelink = False

# Exclude copy button from appearing over notebook cell numbers by using :not()
# The default copybutton selector is `div.highlight pre`
# https://github.com/executablebooks/sphinx-copybutton/blob/master/sphinx_copybutton/__init__.py#L82
copybutton_selector = ":not(.prompt) > div.highlight pre"

# -- Options for EPUB output -------------------------------------------------

epub_exclude_files = ["search.html"]
