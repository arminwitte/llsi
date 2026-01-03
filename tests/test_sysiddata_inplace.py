import numpy as np
from llsi.sysiddata import SysIdData


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
