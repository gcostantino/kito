"""Sphinx configuration for the Kito documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

from kito import __version__  # noqa: E402

project = 'Kito'
copyright = '2026, Giuseppe Costantino'
author = 'Giuseppe Costantino'
release = __version__
version = '.'.join(release.split('.')[:2])

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

myst_heading_anchors = 3

exclude_patterns = ['_build']
html_theme = 'sphinx_rtd_theme'

autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = False
