import pandas as pd
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
import numpy as np

from ysfitsutilpy import filemgmt

class TestFileMgmt:
    """Tests for filemgmt module."""

    def test_make_summary(self, tmp_path):
        """Test creating summary table from FITS files."""
        # Create files with headers
        keys = ["OBJECT", "FILTER", "EXPTIME"]
        data = [
            ("M1", "V", 10.0),
            ("M1", "B", 20.0),
            ("M2", "V", 10.0)
        ]

        paths = []
        for i, (obj, filt, exp) in enumerate(data):
            p = tmp_path / f"img{i}.fits"
            hdr = fits.Header()
            hdr["OBJECT"] = obj
            hdr["FILTER"] = filt
            hdr["EXPTIME"] = exp
            fits.writeto(p, np.zeros((10, 10)), header=hdr)
            paths.append(str(p))

        # Run make_summary
        df = filemgmt.make_summary(
            paths,
            keywords=keys,
            verbose=False
        )

        assert len(df) == 3
        assert "file" in df.columns
        assert list(df["OBJECT"]) == ["M1", "M1", "M2"]
        assert list(df["FILTER"]) == ["V", "B", "V"]
        np.testing.assert_allclose(df["EXPTIME"], [10.0, 20.0, 10.0])

    def test_load_if_exists(self, tmp_path):
        """Test load_if_exists functionality."""
        p = tmp_path / "test.fits"

        # Test non-existent
        res = filemgmt.load_if_exists(p, loader=fits.open)
        assert res is None

        # Test existent
        fits.writeto(p, np.zeros((10,10)))
        res = filemgmt.load_if_exists(p, loader=fits.open)
        assert isinstance(res, (fits.HDUList, list)) # load_if_exists returns HDUList by default from fits.open
