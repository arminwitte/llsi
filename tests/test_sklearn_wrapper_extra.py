import numpy as np
import pytest
from llsi.sklearn import LTIModel


def test_predict_before_fit_raises():
    clf = LTIModel()
    with pytest.raises(RuntimeError):
        clf.predict(np.zeros((3, 1)))


def test_fit_shape_mismatch_raises():
    X = np.zeros((5, 1))
    y = np.zeros((4, 1))
    clf = LTIModel()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_fit_with_1d_input_monkeypatch_sysid():
    class DummyModel:
        def simulate(self, u):
            return np.ones((u.shape[1], 1)) * 3.0

    import llsi.sklearn as sk

    orig_sysid = getattr(sk, "sysid", None)

    def fake_sysid(data, y_names, u_names, order, method=None, settings=None):
        return DummyModel()

    sk.sysid = fake_sysid

    try:
        X = np.arange(6)
        y = np.arange(6)
        clf = LTIModel(method="n4sid", order=1, Ts=1.0)
        clf.fit(X, y)
        yp = clf.predict(X)
        assert yp.shape == (6, 1)
        assert np.allclose(yp, 3.0)
    finally:
        if orig_sysid is None:
            delattr(sk, "sysid")
        else:
            sk.sysid = orig_sysid
