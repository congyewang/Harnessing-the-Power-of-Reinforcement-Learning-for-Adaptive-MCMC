import os
import sys

# Specify the Path of the Source Code
sys.path.insert(0, os.path.abspath("../../src/pyrlmala/"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyrlmala"
copyright = "2024, Congye Wang"
author = "Congye Wang"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Automatically generates documents from docstring
    'sphinx.ext.napoleon',          # Support for Google and NumPy style docstring
    'sphinx.ext.autosummary',       # Automatic generation of overviews
    'sphinx_autodoc_typehints',     # Support for type annotations
    'sphinx_markdown_builder',      # Support for Markdown
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for autodoc -----------------------------------------------------
autodoc_inherit_docstrings = True
