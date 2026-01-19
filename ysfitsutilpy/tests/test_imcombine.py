import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units as u

from ysfitsutilpy.imutil import imcombine, ndcombine

class TestNDCombine:
    """Tests for ndcombine function (core algorithmic logic)."""

    def test_basic_average(self):
        """Test simple average combination."""
        # 3 images, 10x10, values 1, 2, 3.
        # Average should be 2.
        arr = np.zeros((3, 10, 10))
        arr[0] += 1
        arr[1] += 2
        arr[2] += 3

        combined = ndcombine(arr, combine="average")
        np.testing.assert_allclose(combined, 2.0, rtol=1e-6)

    def test_basic_median(self):
        """Test simple median combination."""
        arr = np.zeros((3, 10, 10))
        arr[0] += 1
        arr[1] += 10
        arr[2] += 100

        # Median is 10.
        combined = ndcombine(arr, combine="median")
        np.testing.assert_allclose(combined, 10.0, rtol=1e-6)

    def test_sigma_clip(self):
        """Test sigma clipping."""
        # 5 images. 4 have value 10, 1 has value 100 (outlier).
        arr = np.ones((5, 10, 10)) * 10.0
        arr[4] = 100.0

        # Sigma clip with sigma=3.
        # Mean ~ 28. Std ~ 36.
        # 100 is (100-28)/36 = 2 sigma...
        # Wait, if we use sample std?
        # Let's make it more extreme.
        arr[4] = 1000.0

        # combine="average", reject="sigclip", sigma=[3, 3]
        combined = ndcombine(
            arr,
            combine="average",
            reject="sigclip",
            sigma=[1.0, 1.0],
            verbose=False
        )

        # Should reject 1000.0 and average the rest (10.0).
        np.testing.assert_allclose(combined, 10.0, rtol=1e-6)

    def test_minmax_clip(self):
        """Test minmax rejection."""
        # 0, 10, 10, 10, 100
        arr = np.array([0, 10, 10, 10, 100])
        # Reshape to (N, 1, 1) needed for ndcombine?
        # ndcombine expects (N, y, x).
        arr = arr[:, None, None] * np.ones((5, 2, 2))

        # nlow=1, nhigh=1 -> reject lowest and highest.
        combined = ndcombine(
            arr,
            combine="average",
            reject="minmax",
            n_minmax=[1, 1]
        )

        # Remaining: 10, 10, 10 -> Average 10.
        np.testing.assert_allclose(combined, 10.0, rtol=1e-6)


class TestImCombine:
    """Tests for imcombine wrapper with FITS files."""

    def test_imcombine_files(self, tmp_path):
        """Test combining FITS files."""
        # Create 3 files
        vals = [10.0, 20.0, 30.0]
        paths = []
        for i, v in enumerate(vals):
            d = np.ones((10, 10)) * v
            p = tmp_path / f"test_{i}.fits"
            CCDData(d, unit="adu").write(p)
            paths.append(p)

        # Combine
        outpath = tmp_path / "combined.fits"

        res = imcombine(
            paths,
            output=outpath,
            combine="average",
            reject="none"
        )

        # Check result object
        np.testing.assert_allclose(res.data, 20.0, rtol=1e-6)

        # Check file
        loaded = CCDData.read(outpath)
        np.testing.assert_allclose(loaded.data, 20.0, rtol=1e-6)
