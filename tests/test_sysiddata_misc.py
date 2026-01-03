import numpy as np
import pytest
from llsi.sysiddata import SysIdData


def test_add_remove_and_time():
    data = SysIdData(Ts=1.0, y=np.arange(5.0), u=np.arange(5.0) * 2)
    assert data.N == 5
    t = data.time()
    assert np.allclose(t, np.arange(5.0))

    data.add_series(z=np.ones(5))
    assert 'z' in data.series
    data.remove('z')
    assert 'z' not in data.series


def test_equidistant_and_center_copy():
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
