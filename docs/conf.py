# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

mpl_config_dir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "_build", "mplconfig"
)
os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
os.makedirs(mpl_config_dir, exist_ok=True)

project = "Hypergraphx"

import datetime

year = datetime.datetime.now().year

copyright = f"{year}, HGX-Team"
author = "HGX-Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static", "../assets"]
html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css",
]

autodoc_typehints = "none"

suppress_warnings = ["ref.python"]

# Render notebooks without executing them on RTD.
nbsphinx_execute = "never"

# Global substitutions for RST files.
rst_epilog = f"""
.. |copyright| replace:: {copyright}
"""
