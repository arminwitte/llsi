import numpy as np
import pytest

from llsi.matlab import arx, compare, iddata, impulse, n4sid, oe, pem, step


def test_matlab_api(monkeypatch):
    # Mock matplotlib
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("matplotlib.pyplot.figure", lambda: None)
    monkeypatch.setattr("matplotlib.pyplot.plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.subplot", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.legend", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.xlabel", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.ylabel", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.title", lambda *args, **kwargs: None)

    # Generate data
    N = 100
    Ts = 0.1
    t = np.arange(N) * Ts
    u = np.sin(t)
    y = 0.5 * u + 0.1 * np.random.randn(N)  # Simple static relation + noise

    # iddata
    data = iddata(y, u, Ts)
    assert data.Ts == Ts
    # Check names
    assert "y" in data.series
    assert "u" in data.series

    # arx
    sys_arx = arx(data, [1, 1, 0])
    assert sys_arx is not None

    # n4sid
    sys_n4sid = n4sid(data, 1)
    assert sys_n4sid is not None

    # oe
    sys_oe = oe(data, [1, 1, 0])
    assert sys_oe is not None

    # pem
    sys_pem_poly = pem(data, [1, 1, 0])
    assert sys_pem_poly is not None

    sys_pem_ss = pem(data, 1)
    assert sys_pem_ss is not None

    # compare
    fit = compare(data, sys_arx)
    assert isinstance(fit, float)

    # step
    y_step, t_step = step(sys_arx)
    assert len(y_step) == 100

    # impulse
    y_imp, t_imp = impulse(sys_arx)
    assert len(y_imp) == 100


def test_matlab_mimo(monkeypatch):
    # Mock matplotlib
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("matplotlib.pyplot.figure", lambda: None)
    monkeypatch.setattr("matplotlib.pyplot.plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.subplot", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.legend", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.xlabel", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.ylabel", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.title", lambda *args, **kwargs: None)

    N = 100
    Ts = 0.1
    u = np.random.randn(N, 2)
    y = np.random.randn(N, 2)

    data = iddata(y, u, Ts)
    assert "y1" in data.series
    assert "y2" in data.series
    assert "u1" in data.series
    assert "u2" in data.series

    # n4sid supports MIMO? No, llsi implementation raises NotImplementedError
    with pytest.raises(NotImplementedError):
        n4sid(data, 2)

    # Create a MIMO model using sysid directly (po-moesp is default)
    from llsi import sysid

    y_names = [k for k in data.series.keys() if k.startswith("y")]
    u_names = [k for k in data.series.keys() if k.startswith("u")]
    sys = sysid(data, y_names, u_names, 2, method="po-moesp")

    with pytest.MonkeyPatch.context() as m:
        m.setattr("matplotlib.pyplot.show", lambda: None)
        m.setattr("matplotlib.pyplot.figure", lambda: None)
        m.setattr("matplotlib.pyplot.plot", lambda *args, **kwargs: None)
        m.setattr("matplotlib.pyplot.subplot", lambda *args, **kwargs: None)
        m.setattr("matplotlib.pyplot.legend", lambda *args, **kwargs: None)
        m.setattr("matplotlib.pyplot.xlabel", lambda *args, **kwargs: None)
        m.setattr("matplotlib.pyplot.ylabel", lambda *args, **kwargs: None)
        m.setattr("matplotlib.pyplot.title", lambda *args, **kwargs: None)

        compare(data, sys)
        step(sys)
        impulse(sys)
