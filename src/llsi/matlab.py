import matplotlib.pyplot as plt
import numpy as np

from .sysidalg import sysid
from .sysiddata import SysIdData


def iddata(y, u, Ts):
    """
    Create a SysIdData object from output y and input u with sampling time Ts.
    Mimics MATLAB's iddata.
    """
    y = np.array(y)
    u = np.array(u)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    data = SysIdData(Ts=Ts)

    ny = y.shape[1]
    nu = u.shape[1]

    for i in range(ny):
        name = "y" if ny == 1 else f"y{i + 1}"
        data.add_series(**{name: y[:, i]})

    for i in range(nu):
        name = "u" if nu == 1 else f"u{i + 1}"
        data.add_series(**{name: u[:, i]})

    return data


def _get_names(data):
    keys = list(data.series.keys())
    y_names = sorted([k for k in keys if k.startswith("y")])
    u_names = sorted([k for k in keys if k.startswith("u")])
    return y_names, u_names


def arx(data, order):
    """
    Estimate ARX model.
    order: [na nb nk] for SISO
    """
    y_names, u_names = _get_names(data)
    return sysid(data, y_names, u_names, tuple(order), method="arx")


def n4sid(data, order):
    """
    Estimate State-Space model using N4SID.
    order: number of states
    """
    y_names, u_names = _get_names(data)
    return sysid(data, y_names, u_names, order, method="n4sid")


def oe(data, order):
    """
    Estimate Output-Error model.
    order: [nb nf nk]
    """
    nb, nf, nk = order
    y_names, u_names = _get_names(data)
    # Map to ARX structure (na, nb, nk) where na=nf
    return sysid(data, y_names, u_names, (nf, nb, nk), method="oe")


def pem(data, order=None):
    """
    Estimate model using Prediction Error Method.
    """
    y_names, u_names = _get_names(data)
    if isinstance(order, int):
        # State space PEM
        return sysid(data, y_names, u_names, order, method="pem", settings={"init": "n4sid"})
    else:
        # Polynomial PEM (default init is arx)
        return sysid(data, y_names, u_names, tuple(order), method="pem")


def compare(data, model):
    """
    Compare measured output with simulated output.
    """
    y_names, u_names = _get_names(data)

    u_list = [data[name] for name in u_names]
    u = np.column_stack(u_list)

    y_list = [data[name] for name in y_names]
    y = np.column_stack(y_list)

    y_hat = model.simulate(u)
    fit = model.compare(y, u)

    t = np.arange(y.shape[0]) * data.Ts

    plt.figure()
    if y.shape[1] == 1:
        plt.plot(t, y, "k", label="Measured")
        plt.plot(t, y_hat, "b", label=f"Simulated (fit: {fit:.2%})")
        plt.legend()
    else:
        ny = y.shape[1]
        for i in range(ny):
            plt.subplot(ny, 1, i + 1)
            plt.plot(t, y[:, i], "k", label="Measured")
            plt.plot(t, y_hat[:, i], "b", label="Simulated")
            plt.ylabel(y_names[i])
            if i == 0:
                plt.legend()

    plt.xlabel("Time")
    plt.show()

    return fit


def step(model, Tfinal=None):
    """
    Plot step response.
    """
    N = 100
    if Tfinal is not None:
        N = int(Tfinal / model.Ts) + 1

    t, y = model.step_response(N=N)

    plt.figure()
    if y.ndim == 1:
        plt.plot(t, y)
    else:
        plt.plot(t, y)

    plt.title("Step Response")
    plt.show()
    return y, t


def impulse(model, Tfinal=None):
    """
    Plot impulse response.
    """
    N = 100
    if Tfinal is not None:
        N = int(Tfinal / model.Ts) + 1

    t, y = model.impulse_response(N=N)
    plt.figure()
    plt.plot(t, y)
    plt.title("Impulse Response")
    plt.show()
    return y, t
