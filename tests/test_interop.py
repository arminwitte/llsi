import numpy as np
import pytest

from llsi import SysIdData

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_to_pandas():
    t = np.linspace(0, 10, 11)  # 0 to 10, 11 points -> Ts=1
    y = np.sin(t)
    u = np.cos(t)

    data = SysIdData(t=t, y=y, u=u)

    df = data.to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert "y" in df.columns
    assert "u" in df.columns
    assert np.allclose(df.index.values, t)
    assert np.allclose(df["y"].values, y)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_from_pandas_numeric_index():
    df = pd.DataFrame({"y": np.random.randn(10), "u": np.random.randn(10)}, index=np.arange(10) * 0.5)

    data = SysIdData.from_pandas(df)

    assert data.Ts == 0.5
    assert data.N == 10
    assert "y" in data.series
    assert "u" in data.series
    assert np.allclose(data["y"], df["y"].values)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_from_pandas_datetime_index():
    dt_index = pd.date_range(start="2021-01-01", periods=10, freq="100ms")  # 0.1s
    df = pd.DataFrame({"y": np.random.randn(10)}, index=dt_index)

    data = SysIdData.from_pandas(df)

    assert np.isclose(data.Ts, 0.1)
    assert data.N == 10
    # t_start should be 0 relative time? Or we don't care?
    # Implementation choice: convert to relative seconds or keep absolute?
    # SysIdData usually works with relative time (floats).


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_from_pandas_time_col():
    df = pd.DataFrame({"time": np.arange(10), "y": np.random.randn(10)})

    data = SysIdData.from_pandas(df, time_col="time")

    assert data.Ts == 1.0
    assert "y" in data.series
    assert "time" not in data.series  # Should be consumed
