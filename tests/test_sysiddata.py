#!/usr/bin/env python3
"""
Created on Sun Apr  4 20:25:46 2021

@author: armin
"""

import numpy as np
import pytest

from llsi import math as llsi_math
from llsi.sysiddata import SysIdData


@pytest.fixture
def data():
    data = SysIdData(y=[1, 2, 3, 4, 5, 6], u=[1, 4, 9, 16, 25, 36], t=[1, 1.5, 2.5, 3.0, 3.3, 4.1])
    return data


def test_init():
    data = SysIdData(y=[1, 2, 3, 4, 5, 6], u=[1, 4, 9, 16, 25, 36], t=[1, 1.5, 2.5, 3.0, 3.3, 4.1])
    np.testing.assert_equal(data["y"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_equal(data["u"], [1, 4, 9, 16, 25, 36])
    np.testing.assert_equal(data.time, [1, 1.5, 2.5, 3.0, 3.3, 4.1])


def test_equidistant(data):
    data.equidistant()
    np.testing.assert_allclose(data["y"], [1.0, 2.12, 2.74, 3.72, 5.225, 6.0])
    np.testing.assert_allclose(data["u"], [1.0, 4.6, 7.7, 14.04, 27.475, 36.0])
    np.testing.assert_allclose(data.time, [1.0, 1.62, 2.24, 2.86, 3.48, 4.1])


def test_center(data):
    data.equidistant()
    data.center()
    np.testing.assert_allclose(data["y"], [-2.4675, -1.3475, -0.7275, 0.2525, 1.7575, 2.5325])
    np.testing.assert_allclose(
        data["u"],
        [-14.135833, -10.535833, -7.435833, -1.095833, 12.339167, 20.864167],
        rtol=1e-6,
    )


def test_crop(data):
    data.crop(1, -1)
    np.testing.assert_equal(data["y"], [2, 3, 4, 5])
    np.testing.assert_equal(data["u"], [4, 9, 16, 25])
    np.testing.assert_equal(data.time, [1.5, 2.5, 3.0, 3.3])


def test_generate_prbs():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    np.testing.assert_allclose(t[29:32], [29.0, 30.0, 31.0])
    # Check that values are binary (0 or 1)
    assert np.all(np.isin(u, [0.0, 1.0]))
    assert len(u) == 1000


def test_downsample():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    data = SysIdData(t=t, u=u)
    data.equidistant()
    data.downsample(2)
    assert data.Ts == 2.0
    np.testing.assert_allclose(data.time[14:17], [28.0, 30.0, 32.0])
    # Just verify downsampling works, not specific PRBS values
    assert data.N == 500


def test_lowpass():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    data = SysIdData(t=t, u=u)
    data.equidistant()
    data.lowpass(1, 0.1)
    # Verify lowpass filtering smooths the signal
    assert np.all(data["u"] >= 0.0) and np.all(data["u"] <= 1.0)
    assert np.std(data["u"]) < np.std(u)


def test_split():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    data = SysIdData(t=t, u=u)
    data.equidistant()
    data1, data2 = data.split()
    assert data1.N == 500
    assert data2.N == 500
    # Verify split maintains data integrity
    assert data1["u"][0] in [0.0, 1.0]
    assert data2["u"][0] in [0.0, 1.0]


def test_add_series():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    t, v = SysIdData.generate_prbs(1000, 1.0)
    data = SysIdData(t=t, u=u)
    data.add_series(v=v)
    np.testing.assert_allclose(data["v"], v)


def test_remove():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    t, v = SysIdData.generate_prbs(1000, 1.0)
    data = SysIdData(t=t, u=u, v=v)
    data.remove("v")
    with pytest.raises(KeyError):
        _ = data["v"]


# --- Consolidated tests from other files ---


def test_time_with_Ts_and_empty_series():
    d = SysIdData(Ts=0.5)
    assert d.N == 0
    t = d.time
    assert t.size == 0


def test_equidistant_resample_changes_Ts_and_removes_t():
    t = np.array([0.0, 0.7, 1.3, 2.0])
    d = SysIdData(t=t, y=np.array([0.0, 1.0, 0.0, -1.0]))
    d2 = d.equidistant(N=4, inplace=False)
    assert d2.t is None
    assert d2.Ts is not None


def test_center_inplace_and_copy():
    d = SysIdData(Ts=1.0, y=np.array([1.0, 2.0, 3.0]))
    d_copy = d.center(inplace=False)
    assert d_copy is not d
    d.center(inplace=True)
    assert np.allclose(np.mean(list(d.series.values())[0]), 0.0)


def make_sample_data(N=100, Ts=0.1):
    t = None
    Ts = Ts
    series = {
        "u": np.linspace(0, 1, N),
        "y": np.sin(np.linspace(0, 2 * np.pi, N)),
    }
    return SysIdData(series=series, t=t, Ts=Ts)


def test_center_inplace_default_is_false_and_returns_copy():
    d = make_sample_data()
    d0 = d.series["y"].copy()
    d2 = d.center(inplace=False)
    # explicit inplace=False, so original should be unchanged
    assert np.allclose(d.series["y"], d0)
    # returned object should have mean ~0
    assert abs(np.mean(d2.series["y"])) < 1e-12


def test_equidistant_vectorized_and_inplace_false():
    N = 50
    t = np.sort(np.random.rand(N))
    series = {"a": np.sin(t), "b": np.cos(t)}
    d = SysIdData(series=series, t=t, Ts=None)
    d2 = d.equidistant(100, inplace=False)
    # original remains non-equidistant
    assert d.t is not None
    # returned has Ts set and t is None (equidistant)
    assert d2.t is None
    assert d2.Ts is not None


def test_crop_and_chain_behavior():
    d = make_sample_data(N=20)
    d2 = d.crop(end=10, inplace=False)
    assert d.N == 20
    assert d2.N == 10
    # chaining, request copies explicitly
    d3 = d.crop(end=10, inplace=False).center(inplace=False)
    assert d3.N == 10


def test_lowpass_inplace_false_and_returns_copy():
    d = make_sample_data(N=200, Ts=0.01)
    d2 = d.lowpass(order=2, corner_frequency=2.0, inplace=False)
    # original unchanged
    assert np.allclose(d.series["u"], np.linspace(0, 1, d.N))
    # returned has altered data
    assert not np.allclose(d2.series["u"], d.series["u"])


def test_generate_prbs_length_and_time():
    t, u = SysIdData.generate_prbs(128, 0.1, seed=7)
    assert t.shape[0] == 128
    assert u.shape[0] == 128
    assert np.all(np.isin(u, [0.0, 1.0]))


def test_add_remove_and_time_misc():
    data = SysIdData(Ts=1.0, y=np.arange(5.0), u=np.arange(5.0) * 2)
    assert data.N == 5
    t = data.time
    assert np.allclose(t, np.arange(5.0))

    data.add_series(z=np.ones(5))
    assert "z" in data.series
    data.remove("z")
    assert "z" not in data.series


def test_equidistant_and_center_copy_misc():
    t = np.array([0.0, 0.5, 1.5, 2.0])
    data = SysIdData(t=t, y=np.array([0.0, 1.0, 0.0, -1.0]))
    # equidistant with new N
    d2 = data.equidistant(N=4, inplace=False)
    assert d2 is not data
    assert d2.Ts is not None
    # center should return copy when inplace=False
    d3 = d2.center(inplace=False)
    assert d3 is not d2
    assert np.allclose(np.mean(list(d3.series.values())[0]), 0.0)


def test_lowpass_success_with_Ts():
    # create a longer signal and apply lowpass
    d = SysIdData(Ts=0.1, y=np.linspace(0, 1, 200))
    d2 = d.lowpass(order=2, corner_frequency=0.5, inplace=False)
    assert d2.N == d.N
    assert np.all(np.isfinite(list(d2.series.values())[0]))


def test_equidistant_warning_on_downsample(caplog):
    t = np.linspace(0, 1, 50)
    d = SysIdData(t=t, y=np.sin(2 * np.pi * t))
    # request fewer points to trigger the warning
    with caplog.at_level("WARNING"):
        _ = d.equidistant(N=10, inplace=False)
    assert any("Downsampling without filter" in rec.message for rec in caplog.records)


def test_prbs_deterministic_for_seed():
    t1, u1 = SysIdData.generate_prbs(32, Ts=0.1, seed=123)
    t2, u2 = SysIdData.generate_prbs(32, Ts=0.1, seed=123)
    assert np.array_equal(u1, u2)


def test_pandas_datetime_roundtrip_if_available():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pytest.skip("pandas not installed")

    # build a datetime index
    idx = pd.date_range(start="2020-01-01", periods=5, freq="s")
    df = pd.DataFrame({"y": np.arange(5.0)}, index=idx)
    d = SysIdData.from_pandas(df)
    df2 = d.to_pandas()
    # ensure index is numeric (seconds) or datetime-like and data preserved
    assert d.N == 5
    assert np.allclose(df2["y"].values, df["y"].values)


def test_prbs_generation_length_and_values_ops():
    t, u = SysIdData.generate_prbs(16, Ts=0.1, seed=7)
    assert t.shape == (16,)
    assert u.shape == (16,)
    # values should be 0.0 or 1.0
    assert set(np.unique(u)).issubset({0.0, 1.0})


def test_prbs_helpers_return_ints():
    # prbs31 should accept an int and return int
    v = llsi_math.prbs31(1)
    assert isinstance(v, int)


def test_lowpass_requires_Ts():
    # provide a time vector but no Ts -> should raise when calling lowpass
    d = SysIdData(t=np.array([0.0, 0.1, 0.2]), y=np.array([1.0, 2.0, 3.0]))
    assert d.Ts is None
    with pytest.raises(ValueError):
        d.lowpass(order=2, corner_frequency=0.1)


def test_resample_and_downsample_roundtrip():
    # use a longer signal so scipy.decimate has enough padding length
    d = SysIdData(Ts=1.0, y=np.arange(64.0))
    d2 = d.resample(0.5, inplace=False)
    assert d2.N == 32
    d3 = d.resample(2.0, inplace=False)
    assert d3.N == 128
    d4 = d.downsample(2, inplace=False)
    assert d4.N == 32


def test_to_from_pandas_roundtrip_if_available_ops():
    try:
        pass  # type: ignore
    except Exception:
        pytest.skip("pandas not installed")

    d = SysIdData(Ts=0.5, y=np.array([0.1, 0.2, 0.3, 0.4]))
    df = d.to_pandas()
    assert hasattr(df, "index")
    d2 = SysIdData.from_pandas(df)
    assert d2.N == d.N


def test_equidistant_interpolation_method_string():
    """Test equidistant with single interpolation method applied to all series."""
    # Create non-equidistant data with a step function
    t = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    u = np.array([0.0, 1.0, 1.0, 0.5, 0.5])  # Step-like input
    y = np.array([0.0, 0.2, 0.5, 0.7, 0.9])  # Smooth output

    # Test linear interpolation
    d_linear = SysIdData(t=t.copy(), u=u.copy(), y=y.copy())
    d_linear.equidistant(N=20, method="linear", inplace=True)
    assert d_linear.t is None  # Equidistant has no time vector
    assert d_linear.N == 20

    # Test previous (ZOH) interpolation
    d_prev = SysIdData(t=t.copy(), u=u.copy(), y=y.copy())
    d_prev.equidistant(N=20, method="previous", inplace=True)
    assert d_prev.t is None  # Equidistant has no time vector
    assert d_prev.N == 20

    # With ZOH, values should be closer to original (no smoothing)
    # Linear should have some intermediate values
    assert np.any(d_linear.series["u"] != d_prev.series["u"])


def test_equidistant_interpolation_method_dict():
    """Test equidistant with per-series interpolation methods."""
    # Create non-equidistant data
    t = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    u = np.array([0.0, 1.0, 1.0, 0.5, 0.5])  # Step-like input (use "previous")
    y = np.array([0.0, 0.2, 0.5, 0.7, 0.9])  # Smooth output (use "linear")

    # Apply different methods to different series
    d_mixed = SysIdData(t=t.copy(), u=u.copy(), y=y.copy())
    d_mixed.equidistant(N=20, method={"u": "previous", "y": "linear"}, inplace=True)

    assert d_mixed.t is None  # Equidistant has no time vector
    assert d_mixed.N == 20

    # Compare with applying each method separately
    d_u_prev = SysIdData(t=t.copy(), u=u.copy(), y=y.copy())
    d_u_prev.equidistant(N=20, method="previous", inplace=True)

    d_y_linear = SysIdData(t=t.copy(), u=u.copy(), y=y.copy())
    d_y_linear.equidistant(N=20, method="linear", inplace=True)

    # u series should match the "previous" version
    np.testing.assert_allclose(d_mixed.series["u"], d_u_prev.series["u"], rtol=1e-10)

    # y series should match the "linear" version
    np.testing.assert_allclose(d_mixed.series["y"], d_y_linear.series["y"], rtol=1e-10)


def test_repr_equidistant():
    """Test __repr__ for equidistant data."""
    d = SysIdData(Ts=0.1, u=np.array([1.0, 2.0, 3.0]), y=np.array([0.1, 0.2, 0.3]))
    repr_str = repr(d)

    assert "SysIdData" in repr_str
    assert "N=3" in repr_str
    assert "Ts=0.1000s" in repr_str
    assert "Time:" in repr_str
    assert "Series:" in repr_str
    assert "u" in repr_str
    assert "y" in repr_str


def test_repr_non_equidistant():
    """Test __repr__ for non-equidistant data."""
    t = np.array([0.0, 0.5, 2.0])
    d = SysIdData(t=t, u=np.array([1.0, 2.0, 3.0]))
    repr_str = repr(d)

    assert "SysIdData" in repr_str
    assert "N=3" in repr_str
    assert "Non-equidistant" in repr_str
    assert "Time:" in repr_str
    assert "Series:" in repr_str
    assert "u" in repr_str


def test_repr_empty():
    """Test __repr__ for empty data."""
    d = SysIdData(Ts=0.1)
    repr_str = repr(d)

    assert "SysIdData" in repr_str
    assert "N=0" in repr_str
    assert "Time: empty" in repr_str
    assert "Series: (none)" in repr_str


def test_differentiate_sine_equidistant():
    """Test differentiation of a sine wave (exact derivative available)."""
    # d/dt[sin(x)] = cos(x), so we can check accuracy
    N = 1001
    t = np.linspace(0, 2 * np.pi, N)
    Ts = t[1] - t[0]
    y = np.sin(t)
    y_expected = np.cos(t)  # Exact derivative

    d = SysIdData(Ts=Ts, y=y)
    d.differentiate("y", inplace=True)

    # Check that derivative series exists
    assert "dy" in d.series
    assert d.series["dy"].shape == d.series["y"].shape

    # Check accuracy: central difference should be quite accurate for smooth functions
    # Allow larger tolerance at boundaries (edge_order=2 is less accurate there)
    np.testing.assert_allclose(d.series["dy"][10:-10], y_expected[10:-10], atol=1e-2)


def test_differentiate_polynomial_equidistant():
    """Test differentiation of a polynomial (exact derivative available)."""
    # y = x^2, dy/dx = 2x
    N = 101
    x = np.linspace(0, 10, N)
    Ts = x[1] - x[0]
    y = x**2
    y_expected = 2 * x

    d = SysIdData(Ts=Ts, y=y)
    d.differentiate("y", new_key="dy", inplace=True)

    assert "dy" in d.series
    # Polynomial differentiation should be very accurate with central differences
    np.testing.assert_allclose(d.series["dy"][5:-5], y_expected[5:-5], atol=1e-10)


def test_differentiate_custom_key():
    """Test differentiation with custom output key name."""
    t = np.linspace(0, 1, 51)
    Ts = t[1] - t[0]
    position = np.sin(2 * np.pi * t)

    d = SysIdData(Ts=Ts, position=position)
    d.differentiate("position", new_key="velocity", inplace=True)

    assert "velocity" in d.series
    assert "position" in d.series
    assert "dposition" not in d.series  # Should use custom key, not default


def test_differentiate_not_inplace():
    """Test differentiation with inplace=False returns a copy."""
    t = np.linspace(0, 1, 51)
    Ts = t[1] - t[0]
    y = t**2

    d = SysIdData(Ts=Ts, y=y)
    d_copy = d.differentiate("y", inplace=False)

    # Original should not have derivative
    assert "dy" not in d.series
    # Copy should have derivative
    assert "dy" in d_copy.series
    # Original y should be unchanged
    np.testing.assert_array_equal(d.series["y"], y)


def test_differentiate_non_equidistant():
    """Test differentiation on non-equidistant data."""
    # Non-equidistant time points
    t = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0])
    # Simple linear function: y = 2*t + 1, dy/dt = 2
    y = 2 * t + 1

    d = SysIdData(t=t, y=y)
    d.differentiate("y", inplace=True)

    assert "dy" in d.series
    assert d.series["dy"].shape == d.series["y"].shape
    # Derivative of linear function should be approximately constant
    np.testing.assert_allclose(d.series["dy"], 2.0, atol=1e-10)


def test_differentiate_nonexistent_key():
    """Test that differentiation of nonexistent key raises error."""
    d = SysIdData(Ts=0.1, y=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="Series 'z' not found"):
        d.differentiate("z")


def test_differentiate_multiple_series():
    """Test differentiating different series independently."""
    t = np.linspace(0, 1, 51)
    Ts = t[1] - t[0]
    u = np.sin(2 * np.pi * t)  # Control input
    y = 0.8 * np.sin(2 * np.pi * t - 0.5)  # Measurement

    d = SysIdData(Ts=Ts, u=u, y=y)
    d.differentiate("u", new_key="u_dot", inplace=True)
    d.differentiate("y", new_key="y_dot", inplace=True)

    assert "u_dot" in d.series
    assert "y_dot" in d.series
    assert len(d.series) == 4  # u, y, u_dot, y_dot

    # Check that derivatives have correct shapes
    assert d.series["u_dot"].shape == d.series["u"].shape
    assert d.series["y_dot"].shape == d.series["y"].shape
