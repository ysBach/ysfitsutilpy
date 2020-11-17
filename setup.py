"""
Convenience utilities made by ysBach especially for dealing FITS files in astronomical sciences.
"""

from setuptools import setup, find_packages

setup_requires = []
install_requires = [
    'numpy',
    'astropy >= 2.0',
    'ccdproc >= 1.3',
    'fitsio',
    'bottleneck'
]

classifiers = [
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3"
]

setup(
    name="ysfitsutilpy",
    version="0.2.dev",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="",
    license="BSD 3-clause",
    keywords="",
    url="",
    classifiers=classifiers,
    packages=find_packages(),
    entry_points={
            'console_scripts': [
                'pimarith = ysfitsutilpy.imutil.scripts.pimarith:main'
            ]
    },
    python_requires='>=3.6',
    install_requires=install_requires)
