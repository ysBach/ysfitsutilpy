# Sphinx configuration for ysfitsutilpy documentation
import os
import sys

# Add package to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ysfitsutilpy"
copyright = "2024, Yoonsoo P. Bach"
author = "Yoonsoo P. Bach"
release = "0.2"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # "numpydoc",  # Conflict with napoleon
    "sphinx_copybutton",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Numpydoc settings
# numpydoc_show_class_members = False

# Intersphinx mapping to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "photutils": ("https://photutils.readthedocs.io/en/stable/", None),
    "ccdproc": ("https://ccdproc.readthedocs.io/en/latest/", None),
    "astroscrappy": ("https://astroscrappy.readthedocs.io/en/stable/", None),
}

# Default role for backticks
default_role = "obj"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# Suppress warnings for missing references in external packages
nitpicky = False

# Mock imports for modules that might not be available or hard to build
autodoc_mock_imports = [
    "astro_ndslice",
    "astroscrappy",
    "bottleneck",
    "numba",
    "ccdproc",
    "pandas",
]
