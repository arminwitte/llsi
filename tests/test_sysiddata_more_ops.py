import numpy as np
import pytest
from llsi.sysiddata import SysIdData


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
        d2 = d.equidistant(N=10, inplace=False)
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
    idx = pd.date_range(start="2020-01-01", periods=5, freq="S")
    df = pd.DataFrame({"y": np.arange(5.0)}, index=idx)
    d = SysIdData.from_pandas(df)
    df2 = d.to_pandas()
    # ensure index is numeric (seconds) or datetime-like and data preserved
    assert d.N == 5
    assert np.allclose(df2["y"].values, df["y"].values)
