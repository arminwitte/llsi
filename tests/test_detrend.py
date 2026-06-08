"""Tests for SysIdData.detrend() method."""

import numpy as np
import pytest

from llsi import SysIdData


class TestDetrend:
    """Test detrend functionality in SysIdData."""

    def test_detrend_linear(self):
        """Test linear detrending removes linear trend."""
        t = np.linspace(0, 10, 100)
        y = t + np.random.randn(100) * 0.1  # linear trend + noise
        u = np.sin(t)

        data = SysIdData(Ts=0.1, t=t, y=y, u=u)
        result = data.detrend(method="linear", inplace=False)

        # After linear detrend, mean should be close to 0
        assert np.abs(np.mean(result["y"])) < 0.01
        assert np.abs(np.mean(result["u"])) < 0.1  # sin(t) has mean ~0

    def test_detrend_constant(self):
        """Test constant detrending (mean removal)."""
        t = np.linspace(0, 10, 100)
        y = t + 5.0 + np.random.randn(100) * 0.1  # offset + noise
        u = np.sin(t) + 3.0

        data = SysIdData(Ts=0.1, t=t, y=y, u=u)
        result = data.detrend(method="constant", inplace=False)

        # After constant detrend, mean should be ~0
        assert np.abs(np.mean(result["y"])) < 0.01
        assert np.abs(np.mean(result["u"])) < 0.01

    def test_detrend_standardized(self):
        """Test standardized detrending (Z-score normalization)."""
        t = np.linspace(0, 10, 100)
        y = t * 2.0 + 10.0 + np.random.randn(100) * 0.1
        u = np.sin(t) * 3.0 + 5.0

        data = SysIdData(Ts=0.1, t=t, y=y, u=u)
        result = data.detrend(method="standardized", inplace=False)

        # After standardization, mean ~0 and std ~1
        assert np.abs(np.mean(result["y"])) < 0.01
        assert np.abs(np.std(result["y"]) - 1.0) < 0.01
        assert np.abs(np.mean(result["u"])) < 0.01
        assert np.abs(np.std(result["u"]) - 1.0) < 0.01

    def test_detrend_inplace(self):
        """Test inplace=True modifies the original object."""
        t = np.linspace(0, 10, 100)
        y = t + np.random.randn(100) * 0.1

        data = SysIdData(Ts=0.1, t=t, y=y)
        original_y = data["y"].copy()

        result = data.detrend(method="linear", inplace=True)

        # Should return self
        assert result is data
        # Original should be modified
        assert not np.allclose(data["y"], original_y)

    def test_detrend_not_inplace(self):
        """Test inplace=False returns a copy."""
        t = np.linspace(0, 10, 100)
        y = t + np.random.randn(100) * 0.1

        data = SysIdData(Ts=0.1, t=t, y=y)
        original_y = data["y"].copy()

        result = data.detrend(method="linear", inplace=False)

        # Should return a different object
        assert result is not data
        # Original should be unchanged
        assert np.allclose(data["y"], original_y)
        # Result should be detrended
        assert np.abs(np.mean(result["y"])) < 0.01

    def test_detrend_chaining(self):
        """Test that detrend can be chained with other methods."""
        t = np.linspace(0, 10, 100)
        y = t + np.random.randn(100) * 0.1

        data = SysIdData(Ts=0.1, t=t, y=y)
        result = data.detrend(method="linear").center()

        assert result is not None
        assert isinstance(result, SysIdData)

    def test_detrend_invalid_method(self):
        """Test that invalid method raises ValueError."""
        t = np.linspace(0, 10, 100)
        y = np.random.randn(100)

        data = SysIdData(Ts=0.1, t=t, y=y)

        with pytest.raises(ValueError, match="Invalid detrend method"):
            data.detrend(method="invalid")

    def test_detrend_default_method(self):
        """Test that default method is 'linear'."""
        t = np.linspace(0, 10, 100)
        y = t + np.random.randn(100) * 0.1

        data = SysIdData(Ts=0.1, t=t, y=y)
        result = data.detrend(inplace=False)

        # Should behave like linear detrend
        assert np.abs(np.mean(result["y"])) < 0.01

    def test_detrend_linear_stores_params(self):
        """Test that linear detrend stores slope and intercept."""
        t = np.linspace(0, 10, 100)
        y = 2.0 * t + 5.0 + np.random.randn(100) * 0.1

        data = SysIdData(Ts=0.1, t=t, y=y)
        data.detrend(method="linear", inplace=True)

        # Check that parameters are stored
        assert "y" in data.trend_slopes
        assert "y" in data.trend_intercepts
        # Slope should be close to 2.0
        assert np.abs(data.trend_slopes["y"] - 2.0) < 0.1
        # Intercept should be close to 5.0
        assert np.abs(data.trend_intercepts["y"] - 5.0) < 0.5

    def test_unscale_reverses_detrend(self):
        """Test that unscale reverses linear detrending."""
        t = np.linspace(0, 10, 100)
        y_original = 2.0 * t + 5.0 + np.random.randn(100) * 0.1

        data = SysIdData(Ts=0.1, t=t, y=y_original.copy())
        original_y = data["y"].copy()

        # Apply detrend
        data.detrend(method="linear", inplace=True)

        # Reverse with unscale
        data.unscale(inplace=True)

        # Should be back to original
        assert np.allclose(data["y"], original_y)
        assert len(data.trend_slopes) == 0
        assert len(data.trend_intercepts) == 0

    def test_apply_detrend_from(self):
        """Test apply_detrend_from applies same detrending to test data."""
        t = np.linspace(0, 10, 100)

        train_data = SysIdData(Ts=0.1, t=t, y=2.0 * t + 5.0 + np.random.randn(100) * 0.1)
        test_data = SysIdData(Ts=0.1, t=t, y=2.0 * t + 5.0 + np.random.randn(100) * 0.1)

        # Detrend training data
        train_data.detrend(method="linear", inplace=True)

        # Apply same detrending to test data
        test_data.apply_detrend_from(train_data, inplace=True)

        # Check that parameters match
        assert test_data.trend_slopes["y"] == train_data.trend_slopes["y"]
        assert test_data.trend_intercepts["y"] == train_data.trend_intercepts["y"]

        # Both should have similar means after detrending
        assert np.abs(np.mean(train_data["y"])) < 0.1
        assert np.abs(np.mean(test_data["y"])) < 0.1

    def test_detrend_then_unscale_roundtrip(self):
        """Test detrend followed by unscale is a perfect roundtrip."""
        t = np.linspace(0, 10, 100)
        y = 3.0 * t - 2.0 + np.random.randn(100) * 0.01

        data = SysIdData(Ts=0.1, t=t, y=y)
        original = data["y"].copy()

        # Detrend and then unscale
        data.detrend(method="linear", inplace=True).unscale(inplace=True)

        assert np.allclose(data["y"], original)
