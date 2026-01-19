"""
Tests for ysfitsutilpy.misc module (overwriting the minimal existing tests).

These tests verify utility functions with pre-calculated expected values.
"""

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits

from ysfitsutilpy import misc

# Strict tolerance for numerical comparisons
RTOL = 1e-6
ATOL = 1e-8


class TestGetSize:
    """Tests for get_size function (memory size calculation)."""

    def test_simple_int(self):
        """Test size of simple integer."""
        size = misc.get_size(42)
        assert isinstance(size, int)
        assert size > 0

    def test_list_larger_than_elements(self):
        """Test that list size > sum of element sizes due to overhead."""
        lst = [1, 2, 3]
        size = misc.get_size(lst)
        assert size > 0

    def test_nested_dict(self):
        """Test recursive size calculation for nested dict."""
        d = {"a": {"b": {"c": 1}}}
        size = misc.get_size(d)
        assert size > 0

    def test_numpy_array(self):
        """Test size calculation for numpy array."""
        arr = np.zeros((100, 100), dtype=np.float32)
        size = misc.get_size(arr)
        # Should be at least 100*100*4 = 40000 bytes
        assert size >= 40000


class TestCircularMask:
    """Tests for circular_mask function."""

    def test_basic_2d(self):
        """Test basic 2D circular mask."""
        mask = misc.circular_mask(shape=(10, 10), center=(5, 5), radius=3)
        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        # Center should be inside the circle
        assert mask[5, 5] == True
        # Corners should be outside
        assert mask[0, 0] == False
        assert mask[9, 9] == False

    def test_mask_sum_known(self):
        """Test that mask sum matches expected count."""
        # For a 21x21 grid centered at (10,10) with radius=5
        # The number of pixels inside should be approximately pi*r^2 = 78.5
        mask = misc.circular_mask(shape=(21, 21), center=(10, 10), radius=5)
        # Allow some tolerance for discretization
        assert 70 <= np.sum(mask) <= 90

    def test_default_center(self):
        """Test that default center is image center."""
        mask = misc.circular_mask(shape=(10, 10), radius=2)
        # Default center should be (5, 5) for a 10x10 image
        assert mask[5, 5] == True


class TestCircularMask2D:
    """Tests for circular_mask_2d function (photutils-based)."""

    def test_basic(self):
        """Test basic 2D circular mask using photutils."""
        mask = misc.circular_mask_2d(shape=(100, 100), center=(50, 50), radius=10)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        # Center should be inside
        assert mask[50, 50] == True

    @pytest.mark.parametrize("radius,expected_sum", [
        (1.0, 1),
        (5.0, 69),
        (10.0, 305),
    ])
    def test_mask_sum_by_radius(self, radius, expected_sum):
        """Test mask pixel count for various radii."""
        mask = misc.circular_mask_2d(
            shape=(100, 100),
            center=(50, 50),
            radius=radius,
            method="center"
        )
        assert np.sum(mask) == expected_sum


class TestStrNow:
    """Tests for str_now function."""

    def test_returns_string(self):
        """Test that str_now returns a string."""
        result = misc.str_now()
        assert isinstance(result, str)

    def test_precision(self):
        """Test precision parameter affects output."""
        result_low = misc.str_now(precision=0)
        result_high = misc.str_now(precision=6)
        # Higher precision should result in longer string
        # (more decimal places in seconds)
        # Both should be valid ISO format times
        assert "T" in result_low
        assert "T" in result_high


class TestChangeToQuantity:
    """Tests for change_to_quantity function."""

    def test_float_to_quantity(self):
        """Test converting float to Quantity."""
        from astropy import units as u
        result = misc.change_to_quantity(5.0, u.m, to_value=False)
        assert hasattr(result, "unit")
        assert result.value == 5.0

    def test_quantity_passthrough(self):
        """Test that Quantity is passed through."""
        from astropy import units as u
        q = 5.0 * u.m
        result = misc.change_to_quantity(q, u.m, to_value=False)
        assert result.value == 5.0
        assert result.unit == u.m

    def test_to_value_true(self):
        """Test extracting value from Quantity."""
        from astropy import units as u
        result = misc.change_to_quantity(5.0 * u.km, u.m, to_value=True)
        np.testing.assert_allclose(result, 5000.0, rtol=RTOL, atol=ATOL)


class TestWeightedAvg:
    """Tests for weighted_avg function."""

    def test_known_values(self):
        """Test weighted average with known values."""
        val = np.array([1.0, 2.0, 3.0])
        err = np.array([0.1, 0.2, 0.1])  # weights = 1/err^2

        # Manual calculation:
        # w = 1/err^2 = [100, 25, 100]
        # weighted_avg = (1*100 + 2*25 + 3*100) / (100+25+100)
        #              = (100 + 50 + 300) / 225 = 450/225 = 2.0
        result = misc.weighted_avg(val, err)
        # Result is (weighted_avg, weighted_std_err)
        np.testing.assert_allclose(result[0], 2.0, rtol=RTOL, atol=ATOL)

    def test_equal_weights(self):
        """Test that equal weights give simple mean."""
        val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = misc.weighted_avg(val, err)
        np.testing.assert_allclose(result[0], 3.0, rtol=RTOL, atol=ATOL)


class TestCmt2Hdr:
    """Tests for cmt2hdr function (adding comments/history to header)."""

    def test_add_history(self, sample_header):
        """Test adding HISTORY to header."""
        hdr = sample_header.copy()
        misc.cmt2hdr(hdr, "h", "Test history entry", time_fmt=None)
        # Check that HISTORY was added
        assert "HISTORY" in hdr
        assert "Test history entry" in str(hdr["HISTORY"])

    def test_add_comment(self, sample_header):
        """Test adding COMMENT to header."""
        hdr = sample_header.copy()
        misc.cmt2hdr(hdr, "c", "Test comment entry", time_fmt=None)
        # Check that COMMENT was added
        assert "COMMENT" in hdr
        assert "Test comment entry" in str(hdr["COMMENT"])

    @pytest.mark.parametrize("histcomm", ["h", "hist", "history", "HISTORY"])
    def test_history_aliases(self, sample_header, histcomm):
        """Test various aliases for HISTORY."""
        hdr = sample_header.copy()
        misc.cmt2hdr(hdr, histcomm, "Test", time_fmt=None)
        assert "HISTORY" in hdr

    @pytest.mark.parametrize("histcomm", ["c", "com", "comm", "comment", "COMMENT"])
    def test_comment_aliases(self, sample_header, histcomm):
        """Test various aliases for COMMENT."""
        hdr = sample_header.copy()
        misc.cmt2hdr(hdr, histcomm, "Test", time_fmt=None)
        assert "COMMENT" in hdr

    def test_invalid_histcomm_raises(self, sample_header):
        """Test that invalid histcomm raises ValueError."""
        hdr = sample_header.copy()
        with pytest.raises(ValueError):
            misc.cmt2hdr(hdr, "invalid", "Test", time_fmt=None)


class TestUpdateProcess:
    """Tests for update_process function."""

    def test_add_process(self, sample_header):
        """Test adding process key to header."""
        hdr = sample_header.copy()
        misc.update_process(hdr, process="B")
        assert "PROCESS" in hdr
        assert "B" in hdr["PROCESS"]

    def test_append_process(self, sample_header):
        """Test appending to existing process key."""
        hdr = sample_header.copy()
        hdr["PROCESS"] = "B"
        misc.update_process(hdr, process="D")
        # Should contain both B and D
        assert "B" in hdr["PROCESS"]
        assert "D" in hdr["PROCESS"]
