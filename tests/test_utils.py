import numpy as np

from llsi.utils import rise_time, settling_time


class MockModel:
    def __init__(self, t, s):
        self.t = t
        self.s = s

    def step_response(self, N=200):
        return self.t, self.s


def test_rise_time():
    # Create a step response that goes from 0 to 1 linearly over 10 seconds
    t = np.linspace(0, 10, 101)
    s = t / 10.0
    mod = MockModel(t, s)

    # Rise time is time from 10% to 90%
    # 10% is at t=1, 90% is at t=9
    # Rise time should be 8
    rt = rise_time(mod)
    assert np.isclose(rt, 8.0, atol=0.1)


def test_settling_time():
    # Create a step response that settles instantly at t=5
    t = np.linspace(0, 10, 101)
    s = np.zeros_like(t)
    s[t >= 5] = 1.0

    mod = MockModel(t, s)

    # Settling time (default margin 0.01)
    # Should be around 5.0
    st = settling_time(mod)
    assert np.isclose(st, 5.0, atol=0.1)


def test_settling_time_oscillatory():
    # Create a response that goes above 1.01 then settles
    t = np.linspace(0, 10, 101)
    s = np.ones_like(t)
    # At t=2, it spikes to 1.05
    s[np.abs(t - 2) < 0.1] = 1.05
    # At t=8, it is 1.0

    mod = MockModel(t, s)

    # Last time it is outside [0.99, 1.01] is around t=2
    st = settling_time(mod)
    assert np.isclose(st, 2.0, atol=0.2)
