import os
import sys
sys.path.insert(0, os.path.abspath('../../GeoJax'))

project = 'GeoJax'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]
html_theme = 'sphinx_rtd_theme'
