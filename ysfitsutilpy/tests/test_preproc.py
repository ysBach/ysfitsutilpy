import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData

from ysfitsutilpy import preproc


class TestBiasCor:
    """Tests for biascor function."""

    def test_bias_subtraction(self):
        """Test simple bias subtraction."""
        data = np.ones((10, 10)) * 100
        bias = np.ones((10, 10)) * 10
        ccd = CCDData(data, unit="adu")
        mbias = CCDData(bias, unit="adu")

        # Expected: 100 - 10 = 90
        corrected = preproc.biascor(ccd, mbias=mbias)
        np.testing.assert_allclose(corrected.data, 90.0, rtol=1e-6)

    def test_bias_file_input(self, tmp_path):
        """Test bias correction using file path."""
        data = np.ones((5, 5)) * 100
        bias = np.ones((5, 5)) * 10
        ccd = CCDData(data, unit="adu")
        mbias = CCDData(bias, unit="adu")

        bias_path = tmp_path / "mbias.fits"
        mbias.write(bias_path)

        corrected = preproc.biascor(ccd, mbiaspath=bias_path)
        np.testing.assert_allclose(corrected.data, 90.0, rtol=1e-6)


class TestDarkCor:
    """Tests for darkcor function."""

    def test_dark_subtraction_no_scaling(self):
        """Test dark subtraction without exposure scaling."""
        data = np.ones((10, 10)) * 100
        dark = np.ones((10, 10)) * 5
        ccd = CCDData(data, unit="adu")
        # Ensure header has EXPTIME for logic, though not used if dark_scale=False?
        # Function signature: darkcor(..., exptime_key="EXPTIME", dark_scale=False)
        # Actually dark_scale=False is default. It just subtracts.
        mdark = CCDData(dark, unit="adu")

        corrected = preproc.darkcor(ccd, mdark=mdark, dark_scale=False)
        # Expected: 100 - 5 = 95
        np.testing.assert_allclose(corrected.data, 95.0, rtol=1e-6)

    def test_dark_subtraction_with_scaling(self):
        """Test dark subtraction with exposure time scaling."""
        data = np.ones((10, 10)) * 100
        # Dark frame: current=1.0 e/s, but derived from 10s exposure -> value 10?
        # Usually master dark is normalized or we scale it.
        # Let's say master dark is 5 ADU for 5 sec exposure (1 ADU/s).
        # Target image is 10 sec exposure.
        # Expected dark current = 1 ADU/s * 10s = 10 ADU.

        dark_data = np.ones((10, 10)) * 5
        mdark = CCDData(dark_data, unit="adu")
        mdark.header["EXPTIME"] = 5.0

        ccd = CCDData(data, unit="adu")
        ccd.header["EXPTIME"] = 10.0

        # We need to tell the function to scale.
        corrected = preproc.darkcor(
            ccd,
            mdark=mdark,
            dark_scale=True,
            exptime_key="EXPTIME"
        )

        # Expected: 100 - (5 * 10/5) = 100 - 10 = 90
        np.testing.assert_allclose(corrected.data, 90.0, rtol=1e-6)


class TestFlatCor:
    """Tests for flatcor function."""

    def test_flat_correction(self):
        """Test simple flat field correction."""
        data = np.ones((10, 10)) * 100
        # Flat: 0.8 to 1.2
        flat_data = np.ones((10, 10))
        flat_data[:, :5] = 0.8
        flat_data[:, 5:] = 1.2
        mflat = CCDData(flat_data, unit="adu")

        ccd = CCDData(data, unit="adu")

        # By default flatcor divides by mflat.
        # It also has flat_norm_value. Default=1.
        # If flat is normalized (mean ~ 1), we just divide.

        corrected = preproc.flatcor(ccd, mflat=mflat, flat_norm_value=1.0)

        expected_left = 100 / 0.8  # 125
        expected_right = 100 / 1.2 # 83.333...

        np.testing.assert_allclose(corrected.data[:, :5], 125.0, rtol=1e-6)
        np.testing.assert_allclose(corrected.data[:, 5:], 100/1.2, rtol=1e-6)


class TestCCDRed:
    """Tests for the main ccdred wrapper."""

    def test_full_reduction_chain(self, tmp_path):
        """Test bias -> dark -> flat chain via ccdred."""
        # Setup data
        shape = (10, 10)
        # Raw data: 1000 ADU flat background + signal
        raw_val = 1000.0
        ccd = CCDData(np.ones(shape) * raw_val, unit="adu")
        ccd.header["EXPTIME"] = 10.0

        # Bias: constant 10
        bias_val = 10.0
        mbias = CCDData(np.ones(shape) * bias_val, unit="adu")
        mbias_path = tmp_path / "bias.fits"
        mbias.write(mbias_path)

        # Dark: 1 ADU/s -> 10 ADU for 10s. Master dark 5s -> 5 ADU
        dark_rate = 1.0
        mdark_exptime = 5.0
        mdark = CCDData(np.ones(shape) * dark_rate * mdark_exptime, unit="adu")
        mdark.header["EXPTIME"] = mdark_exptime
        mdark_path = tmp_path / "dark.fits"
        mdark.write(mdark_path)

        # Flat: 2.0 everywhere (just to test normalization/division)
        flat_val = 2.0
        mflat = CCDData(np.ones(shape) * flat_val, unit="adu")
        mflat_path = tmp_path / "flat.fits"
        mflat.write(mflat_path)

        # Expected calculation:
        # 1. Bias subtraction: 1000 - 10 = 990
        # 2. Dark subtraction: 990 - (5 * 10/5) = 990 - 10 = 980
        # 3. Flat correction: 980 / 2.0 = 490 (assuming flat not normalized internally or norm_val=1)
        # Note: ccdred has flat_norm_value default=1.

        reduced = preproc.ccdred(
            ccd,
            mbiaspath=mbias_path,
            mdarkpath=mdark_path,
            mflatpath=mflat_path,
            dark_scale=True,
            exptime_dark=5.0,
            flat_norm_value=1.0,
            verbose_bdf=0
        )

        np.testing.assert_allclose(reduced.data, 490.0, rtol=1e-6)
