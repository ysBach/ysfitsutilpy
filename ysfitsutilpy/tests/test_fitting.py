import numpy as np
import pytest
from ysfitsutilpy import fitting

class TestFitting:
    """Tests for fitting module."""

    def test_gaussian1d(self):
        """Test Gaussian 1D fitting with integer grid."""
        # Create data on integer grid 0..49
        x = np.arange(50)
        A, mu, sigma = 10.0, 25.0, 5.0
        y = A * np.exp(-(x - mu)**2 / (2 * sigma**2))

        # fit_model("gaussian1d", data=y)
        # Assuming steps=1 (default) matches x=0,1,2...
        res = fitting.fit_model("gaussian1d", y, amplitude=A, mean=mu, stddev=sigma)

        np.testing.assert_allclose(res.amplitude.value, A, rtol=1e-5)
        np.testing.assert_allclose(res.mean.value, mu, rtol=1e-5)
        np.testing.assert_allclose(res.stddev.value, sigma, rtol=1e-5)

    def test_poly1d(self):
        """Test 1D Polynomial fitting with integer grid."""
        # y = 2x + 1
        x = np.arange(10)
        y = 2 * x + 1

        res = fitting.fit_model("polynomial1d", y, degree=1)

        np.testing.assert_allclose(res.c0.value, 1.0, rtol=1e-5)
        np.testing.assert_allclose(res.c1.value, 2.0, rtol=1e-5)
