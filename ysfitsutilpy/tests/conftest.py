"""
Test configuration and fixtures for ysfitsutilpy tests.

This module provides common fixtures and utilities for testing, including:
- Sample FITS data generation
- CCDData objects with known values
- Temporary directory management
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units as u


# Strict numerical tolerance for physics-based calculations
RTOL = 1e-6
ATOL = 1e-8


@pytest.fixture
def sample_data_2d():
    """2D numpy array with known values for testing."""
    # Reproducible random data with known seed
    rng = np.random.default_rng(42)
    return rng.normal(loc=1000.0, scale=50.0, size=(100, 100)).astype(np.float32)


@pytest.fixture
def sample_data_3d():
    """3D numpy array (stack of images) for testing."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=1000.0, scale=50.0, size=(5, 100, 100)).astype(np.float32)


@pytest.fixture
def sample_header():
    """Sample FITS header with common keywords."""
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 100
    header["NAXIS2"] = 100
    header["OBJECT"] = "TestObject"
    header["EXPTIME"] = 60.0
    header["GAIN"] = 1.5
    header["RDNOISE"] = 5.0
    header["BUNIT"] = "adu"
    header["DATE-OBS"] = "2024-01-15T12:00:00"
    header["RA"] = "12:34:56.78"
    header["DEC"] = "+12:34:56.7"
    return header


@pytest.fixture
def sample_ccddata(sample_data_2d, sample_header):
    """CCDData object with sample data and header."""
    return CCDData(data=sample_data_2d, header=sample_header, unit=u.adu)


@pytest.fixture
def temp_fits_file(sample_data_2d, sample_header, tmp_path):
    """Create a temporary FITS file for testing."""
    fpath = tmp_path / "test_image.fits"
    hdu = fits.PrimaryHDU(data=sample_data_2d, header=sample_header)
    hdu.writeto(fpath, overwrite=True)
    return fpath


@pytest.fixture
def temp_fits_files(sample_data_2d, sample_header, tmp_path):
    """Create multiple temporary FITS files for testing combination."""
    fpaths = []
    rng = np.random.default_rng(42)
    for i in range(5):
        fpath = tmp_path / f"test_image_{i:02d}.fits"
        # Add small offsets to each image
        data = sample_data_2d + rng.normal(0, 10, sample_data_2d.shape)
        hdr = sample_header.copy()
        hdr["FILENAME"] = f"test_image_{i:02d}.fits"
        hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=hdr)
        hdu.writeto(fpath, overwrite=True)
        fpaths.append(fpath)
    return fpaths
