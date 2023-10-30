"""Sphinx configuration file."""

project = "TopoEmbedX"
copyright = "2022-2023, PyT-Team, Inc."
author = "PyT-Team Authors"

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.load_style",
]

# Configure nbsphinx for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"

# To get a prompt similar to the Classic Notebook, use
nbsphinx_input_prompt = " In [%s]:"
nbsphinx_output_prompt = " Out [%s]:"

nbsphinx_allow_errors = True

templates_path = ["_templates"]

source_suffix = [".rst"]

master_doc = "index"

language = "en"

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = None

html_theme = "pydata_sphinx_theme"
html_baseurl = "pyt-team.github.io"
htmlhelp_basename = "pyt-teamdoc"
html_last_updated_fmt = "%c"

latex_elements = {}


latex_documents = [
    (
        master_doc,
        "topoembedx.tex",
        "TopoEmbedX Documentation",
        "PyT-Team",
        "manual",
    ),
]

man_pages = [(master_doc, "topoembedx", "TopoEmbedX Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "topoembedx",
        "TopoEmbedX Documentation",
        author,
        "topoembedx",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]

# configure intersphinx
intersphinx_mapping = {
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "toponetx": ("https://pyt-team.github.io/toponetx/", None),
}

# configure numpydoc
numpydoc_validation_checks = {"all", "GL01", "ES01", "SA01", "EX01"}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
