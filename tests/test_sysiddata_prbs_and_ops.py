import numpy as np
import pytest
from llsi.sysiddata import SysIdData


def test_prbs_generation_length_and_values():
    t, u = SysIdData.generate_prbs(16, Ts=0.1, seed=7)
    assert t.shape == (16,)
    assert u.shape == (16,)
    # values should be 0.0 or 1.0
    assert set(np.unique(u)).issubset({0.0, 1.0})


def test_prbs_helpers_return_ints():
    # prbs31 should accept an int and return int
    v = SysIdData.prbs31(1)
    assert isinstance(v, int)
    v2 = SysIdData.prbs31_fast(1)
    assert isinstance(v2, int)


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


def test_to_from_pandas_roundtrip_if_available():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pytest.skip("pandas not installed")

    d = SysIdData(Ts=0.5, y=np.array([0.1, 0.2, 0.3, 0.4]))
    df = d.to_pandas()
    assert hasattr(df, "index")
    d2 = SysIdData.from_pandas(df)
    assert d2.N == d.N
