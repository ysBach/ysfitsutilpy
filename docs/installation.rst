Installation
============

Requirements
------------

- Python >= 3.8
- numpy >= 1.20
- scipy >= 1.5
- astropy >= 5.0
- ccdproc >= 2.0
- numba
- bottleneck
- astroscrappy


Install from PyPI
-----------------

.. code-block:: bash

   pip install ysfitsutilpy


Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/ysBach/ysfitsutilpy.git
   cd ysfitsutilpy
   pip install .


Optional Dependencies
---------------------

For faster FITS I/O (recommended):

.. code-block:: bash

   pip install fitsio

For faster numerical computations:

.. code-block:: bash

   pip install numexpr
