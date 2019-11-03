"""
Convenience utilities made by ysBach especially for dealing FITS files in astronomical sciences.
"""

from setuptools import setup, find_packages

setup_requires = []
install_requires = ['numpy',
                    'astropy >= 2.0',
                    'ccdproc >= 1.3',
                    'matplotlib >= 2']

classifiers = ["Intended Audience :: Science/Research",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3.6"]

setup(
    name="ysfitsutilpy",
    version="0.1.0.dev",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="",
    license="",
    keywords="",
    url="",
    classifiers=classifiers,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires)
