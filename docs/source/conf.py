# Configuration file for the Sphinx documentation builder.

project = 'Kito'
copyright = '2026, Giuseppe Costantino'
author = 'Giuseppe Costantino'
release = '0.2.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Theme
html_theme = 'sphinx_rtd_theme'  # ReadTheDocs theme

# Paths
templates_path = ['_templates']
exclude_patterns = []
html_static_path = ['_static']

# Napoleon settings (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
