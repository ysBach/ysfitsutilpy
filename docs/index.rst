ysfitsutilpy
============

Convenience utilities for dealing with FITS files in astronomical sciences.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api/index


Installation
------------

.. code-block:: bash

   pip install ysfitsutilpy

Or install from source:

.. code-block:: bash

   git clone https://github.com/ysBach/ysfitsutilpy.git
   cd ysfitsutilpy
   pip install .


Quick Start
-----------

.. code-block:: python

   import ysfitsutilpy as yfu

   # Load a FITS file
   ccd = yfu.load_ccd("image.fits")

   # Combine multiple images
   combined = yfu.imcombine(["img1.fits", "img2.fits", "img3.fits"])


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`