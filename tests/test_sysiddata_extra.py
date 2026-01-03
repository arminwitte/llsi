import numpy as np
import pytest
from llsi.sysiddata import SysIdData


def test_time_with_Ts_and_empty_series():
    d = SysIdData(Ts=0.5)
    assert d.N == 0
    t = d.time()
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
