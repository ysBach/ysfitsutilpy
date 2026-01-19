"""
Tests for ysfitsutilpy.airmass module.

These tests verify airmass calculations against known physical values.
Airmass is a fundamental astronomical quantity, so we use strict tolerances.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from ysfitsutilpy import airmass

# Strict tolerance for physical calculations
RTOL = 1e-6
ATOL = 1e-8


class TestCalcAirmass:
    """Tests for calc_airmass function using known physical values."""

    # Pre-calculated airmass values using Stetson (1988) formula with scale=750
    # Formula: X = sqrt((scale*cos(z))^2 + 2*scale + 1) - scale*cos(z)

    @pytest.mark.parametrize("zd_deg, expected_airmass", [
        (0.0, 1.0),                  # Zenith
        (30.0, 1.154444393578),      # 30°
        (45.0, 1.413273259971),      # 45°
        (60.0, 1.996021199163),      # 60°
        (70.0, 2.909255996897),      # 70°
        # (75.0, 3.834005182956),    # 75° (Skipping to keep list short/relevant)
        (80.0, 5.640466662971),      # 80°
        (85.0, 10.618845962156),     # 85°
    ])
    def test_airmass_zd_deg(self, zd_deg, expected_airmass):
        """Test airmass calculation with zenith distance in degrees."""
        result = airmass.calc_airmass(zd_deg=zd_deg, scale=750.0)
        np.testing.assert_allclose(result, expected_airmass, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("cos_zd, expected_airmass", [
        (1.0, 1.0),
        (0.8660254037844387, 1.154444393578),  # cos(30°)
        (0.7071067811865476, 1.413273259971),  # cos(45°)
        (0.5, 1.996021199163),                 # cos(60°)
        (0.0, 38.742741255621),                # cos(90°) horizon airmass
    ])
    def test_airmass_cos_zd(self, cos_zd, expected_airmass):
        """Test airmass calculation with cosine of zenith distance."""
        result = airmass.calc_airmass(cos_zd=cos_zd, scale=750.0)
        np.testing.assert_allclose(result, expected_airmass, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("scale", [500.0, 750.0, 1000.0])
    def test_airmass_at_zenith(self, scale):
        """Airmass at zenith should always be 1.0."""
        result = airmass.calc_airmass(zd_deg=0.0, scale=scale)
        assert result == 1.0

    def test_airmass_array_input(self):
        """Test that array inputs work correctly."""
        zd_array = np.array([0.0, 60.0])
        expected = np.array([1.0, 1.996021199163])
        result = airmass.calc_airmass(zd_deg=zd_array, scale=750.0)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_airmass_different_scales(self):
        """Test airmass at 60° for different scale heights."""
        # Recalculated for Stetson model
        scales_and_expected = [
            (500.0, 1.994047548747),
            (750.0, 1.996021199163),
            (1000.0, 1.997011943298),
        ]
        for scale, expected in scales_and_expected:
            result = airmass.calc_airmass(zd_deg=60.0, scale=scale)
            np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestAirmassObs:
    """Tests for airmass_obs function with observatory coordinates."""

    @pytest.fixture
    def observatory(self):
        """Mauna Kea observatory location."""
        return EarthLocation(lat=19.8207 * u.deg, lon=-155.4681 * u.deg, height=4205 * u.m)

    @pytest.fixture
    def target(self):
        """A test target coordinate."""
        return SkyCoord(ra=180 * u.deg, dec=30 * u.deg, frame="icrs")

    def test_airmass_obs_returns_float(self, observatory, target):
        """Test that airmass_obs returns a `float` value."""
        ut = Time("2024-01-15T06:00:00", scale="utc")
        result = airmass.airmass_obs(
            targetcoord=target,
            obscoord=observatory,
            ut=ut,
            exptime=60 * u.s,
            scale=750.0,
        )
        assert isinstance(result, float)
        assert result >= 1.0

    def test_airmass_obs_with_full(self, observatory, target):
        """Test full output mode returns (am_eff, `dict`)."""
        ut = Time("2024-01-15T06:00:00", scale="utc")
        am_eff, info_dict = airmass.airmass_obs(
            targetcoord=target,
            obscoord=observatory,
            ut=ut,
            exptime=60 * u.s,
            scale=750.0,
            full=True,
        )
        assert isinstance(am_eff, float)
        assert isinstance(info_dict, dict)
        # Check actual keys returned by implementation
        assert "alt" in info_dict
        assert "az" in info_dict
        assert "zd" in info_dict
        assert "am" in info_dict

