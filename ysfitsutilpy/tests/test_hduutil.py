"""
Tests for ysfitsutilpy.hduutil module.

These tests verify FITS I/O, parsing, and HDU manipulation functions.
"""

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units as u

from ysfitsutilpy import hduutil

# Strict tolerance for numerical comparisons
RTOL = 1e-6
ATOL = 1e-8


class TestParseDataHeader:
    """Tests for _parse_data_header function."""

    def test_parse_ccddata(self, sample_ccddata):
        """Test parsing CCDData object."""
        data, hdr = hduutil._parse_data_header(sample_ccddata)
        np.testing.assert_array_equal(data, sample_ccddata.data)
        assert hdr["OBJECT"] == "TestObject"
        assert hdr["EXPTIME"] == 60.0

    def test_parse_ccddata_no_copy(self, sample_ccddata):
        """Test parsing CCDData without copying (shares memory)."""
        data, hdr = hduutil._parse_data_header(sample_ccddata, copy=False)
        # When copy=False, data should be the same object (views share memory)
        assert np.shares_memory(data, sample_ccddata.data)

    def test_parse_ccddata_with_copy(self, sample_ccddata):
        """Test parsing CCDData with copying (independent memory)."""
        data, hdr = hduutil._parse_data_header(sample_ccddata, copy=True)
        # When copy=True, data should be independent
        assert not np.shares_memory(data, sample_ccddata.data)

    def test_parse_ndarray(self, sample_data_2d):
        """Test parsing plain numpy array."""
        data, hdr = hduutil._parse_data_header(sample_data_2d)
        np.testing.assert_array_equal(data, sample_data_2d)
        assert hdr is None  # ndarray has no header

    def test_parse_fits_file(self, temp_fits_file):
        """Test parsing FITS file path."""
        data, hdr = hduutil._parse_data_header(temp_fits_file)
        assert data.shape == (100, 100)
        assert hdr["OBJECT"] == "TestObject"

    def test_parse_none(self):
        """Test parsing None returns None, None."""
        data, hdr = hduutil._parse_data_header(None)
        assert data is None
        assert hdr is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None, None."""
        data, hdr = hduutil._parse_data_header("")
        assert data is None
        assert hdr is None

    def test_parse_data_only(self, sample_ccddata):
        """Test parsing only data (no header)."""
        data, hdr = hduutil._parse_data_header(
            sample_ccddata, parse_data=True, parse_header=False
        )
        np.testing.assert_array_equal(data, sample_ccddata.data)
        assert hdr is None

    def test_parse_header_only(self, sample_ccddata):
        """Test parsing only header (no data)."""
        data, hdr = hduutil._parse_data_header(
            sample_ccddata, parse_data=False, parse_header=True
        )
        assert data is None
        assert hdr["OBJECT"] == "TestObject"

    def test_parse_number(self):
        """Test parsing a number (float)."""
        data, hdr = hduutil._parse_data_header(42.0)
        assert data == 42.0
        assert hdr is None


class TestInputs2List:
    """Tests for inputs2list function."""

    def test_single_string_path(self, temp_fits_file):
        """Test single string path input."""
        result = hduutil.inputs2list(str(temp_fits_file), path_to_text=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == str(temp_fits_file)

    def test_single_path_object(self, temp_fits_file):
        """Test single Path object input."""
        result = hduutil.inputs2list(temp_fits_file, path_to_text=False)
        assert isinstance(result, list)
        assert len(result) == 1
        # inputs2list currently converts single Path to string via glob.glob
        # We check equality regardless of type (str vs Path)
        assert str(result[0]) == str(temp_fits_file)

    def test_list_of_paths(self, temp_fits_files):
        """Test list of paths input."""
        result = hduutil.inputs2list(temp_fits_files, sort=True)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_glob_pattern(self, temp_fits_files, tmp_path):
        """Test glob pattern input."""
        pattern = str(tmp_path / "test_image_*.fits")
        result = hduutil.inputs2list(pattern, sort=True)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_ccddata_input(self, sample_ccddata):
        """Test CCDData input (passthrough)."""
        result = hduutil.inputs2list(sample_ccddata, accept_ccdlike=True)
        # Function returns list of inputs if accept_ccdlike is True
        assert isinstance(result, list)
        assert result[0] is sample_ccddata

    def test_sort_option(self, temp_fits_files):
        """Test sorting behavior."""
        # Reverse the input list
        reversed_paths = list(reversed(temp_fits_files))
        result = hduutil.inputs2list(reversed_paths, sort=True)
        # Result should be sorted
        assert result == sorted(temp_fits_files)


class TestLoadCcd:
    """Tests for load_ccd function."""

    def test_load_basic(self, temp_fits_file):
        """Test basic FITS loading."""
        ccd = hduutil.load_ccd(temp_fits_file)
        assert isinstance(ccd, CCDData)
        assert ccd.data.shape == (100, 100)
        assert ccd.unit == u.adu

    def test_load_ccddata_false(self, temp_fits_file):
        """Test loading as raw arrays (ccddata=False)."""
        result = hduutil.load_ccd(temp_fits_file, ccddata=False, full=True)
        # Returns (data, var, mask, flags) tuple
        data, var, mask, flags = result
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 100)

    def test_load_with_trimsec(self, temp_fits_file):
        """Test loading with trimsec (section trimming)."""
        # trimsec must be passed; here load_ccd handles it internally using imslice
        # but load_ccd signature is loose; verify it works naturally
        ccd = hduutil.load_ccd(temp_fits_file, trimsec="[11:90,21:80]")
        # Trimsec [11:90,21:80] in FITS notation (1-indexed, x:y)
        # x=11..90 (80 px), y=21..80 (60 px)
        # Python shape (y, x): (60, 80)
        assert ccd.data.shape == (60, 80)


class TestWriteFits:
    """Tests for write2fits function."""

    def test_write_basic(self, sample_data_2d, sample_header, tmp_path):
        """Test basic FITS writing."""
        outpath = tmp_path / "output.fits"
        hduutil.write2fits(sample_data_2d, sample_header, outpath)
        assert outpath.exists()

        # Verify written data
        with fits.open(outpath) as hdul:
            np.testing.assert_array_equal(hdul[0].data, sample_data_2d)
            assert hdul[0].header["OBJECT"] == "TestObject"

    def test_write_return_ccd(self, sample_data_2d, sample_header, tmp_path):
        """Test write2fits with return_ccd=True."""
        outpath = tmp_path / "output.fits"
        result = hduutil.write2fits(
            sample_data_2d, sample_header, outpath, return_ccd=True
        )
        assert isinstance(result, CCDData)
        np.testing.assert_array_equal(result.data, sample_data_2d)


class TestGiveStats:
    """Tests for give_stats function."""

    def test_stats_basic(self, sample_data_2d):
        """Test basic statistics calculation."""
        stats = hduutil.give_stats(sample_data_2d)
        assert isinstance(stats, dict)
        # Check required keys exist (implementation uses avg, madstd, med)
        assert "min" in stats
        assert "max" in stats
        assert "avg" in stats
        assert "med" in stats
        assert "std" in stats

    def test_stats_known_values(self):
        """Test statistics with known values."""
        # Create array with known statistics
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = hduutil.give_stats(arr)

        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        np.testing.assert_allclose(stats["avg"], 3.0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(stats["med"], 3.0, rtol=RTOL, atol=ATOL)
        # std of [1,2,3,4,5] with ddof=0 is sqrt(2), implementation uses ddof=1 for std
        # std of sample [1,2,3,4,5] ddof=1 is sqrt(2.5) ~ 1.5811388
        np.testing.assert_allclose(stats["std"], np.std(arr, ddof=1), rtol=RTOL, atol=ATOL)
