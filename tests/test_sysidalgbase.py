import numpy as np
import pytest

from llsi.ltimodel import LTIModel
from llsi.sysidalgbase import SysIdAlgBase


class ConcreteSysIdAlg(SysIdAlgBase):
    def _ident(self, order):
        pass

    @staticmethod
    def name():
        return "Concrete"


class MockData:
    def __init__(self, data_dict, Ts=1.0):
        self.data = data_dict
        self.Ts = Ts

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data


def test_sysidalgbase_init():
    data_dict = {"y1": np.array([1, 2, 3]), "u1": np.array([4, 5, 6]), "u2": np.array([7, 8, 9])}
    data = MockData(data_dict)

    # Test single input single output
    alg = ConcreteSysIdAlg(data, "y1", "u1", settings={})
    assert alg.y.shape == (3, 1)
    assert alg.u.shape == (3, 1)
    assert np.allclose(alg.y.ravel(), data_dict["y1"])

    # Test multiple inputs
    alg = ConcreteSysIdAlg(data, "y1", ["u1", "u2"], settings={})
    assert alg.u.shape == (3, 2)
    assert np.allclose(alg.u[:, 0], data_dict["u1"])
    assert np.allclose(alg.u[:, 1], data_dict["u2"])


def test_sysidalgbase_init_missing_series():
    """Test that SysIdAlgBase raises helpful errors for missing series."""
    data_dict = {"y1": np.array([1, 2, 3]), "u1": np.array([4, 5, 6])}
    data = MockData(data_dict)

    # Test missing output series
    with pytest.raises(ValueError, match="Output series not found in data: \\['y2'\\]"):
        ConcreteSysIdAlg(data, "y2", "u1", settings={})

    # Test missing input series
    with pytest.raises(ValueError, match="Input series not found in data: \\['u2'\\]"):
        ConcreteSysIdAlg(data, "y1", "u2", settings={})

    # Test multiple missing series
    with pytest.raises(ValueError, match="Input series not found in data: \\['u2', 'u3'\\]"):
        ConcreteSysIdAlg(data, "y1", ["u1", "u2", "u3"], settings={})


def test_sse():
    y = np.array([[1], [2], [3]])
    y_hat = np.array([[1.1], [1.9], [3.0]])

    # Error: -0.1, 0.1, 0.0
    # SSE: 0.01 + 0.01 + 0 = 0.02

    sse = LTIModel.SSE(y - y_hat)
    assert np.isclose(sse, 0.02)
