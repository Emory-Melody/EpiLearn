# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'EpiLearn'
copyright = '2024, exe'
author = 'exe'

# The full version, including alpha/beta/rc tags
release = '1'

# -- General configuration ---------------------------------------------------


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
