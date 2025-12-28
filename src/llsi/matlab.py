"""
MATLAB-like interface for system identification.

This module provides functions that mimic the syntax of the MATLAB System Identification Toolbox,
making it easier for users familiar with MATLAB to use this package.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .ltimodel import LTIModel
from .sysidalg import sysid
from .sysiddata import SysIdData


def iddata(y: Union[np.ndarray, List[float]], u: Union[np.ndarray, List[float]], Ts: float = 1.0) -> SysIdData:
    """
    Create a SysIdData object from output y and input u with sampling time Ts.
    Mimics MATLAB's iddata.

    Args:
        y: Output signal(s).
        u: Input signal(s).
        Ts: Sampling time in seconds.

    Returns:
        SysIdData: The data object.
    """
    y_arr = np.array(y)
    u_arr = np.array(u)

    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(-1, 1)

    data = SysIdData(Ts=Ts)

    ny = y_arr.shape[1]
    nu = u_arr.shape[1]

    for i in range(ny):
        name = "y" if ny == 1 else f"y{i + 1}"
        data.add_series(**{name: y_arr[:, i]})

    for i in range(nu):
        name = "u" if nu == 1 else f"u{i + 1}"
        data.add_series(**{name: u_arr[:, i]})

    return data


def _get_names(data: SysIdData) -> Tuple[List[str], List[str]]:
    """Helper to extract sorted input and output names."""
    keys = list(data.series.keys())
    y_names = sorted([k for k in keys if k.startswith("y")])
    u_names = sorted([k for k in keys if k.startswith("u")])
    return y_names, u_names


def arx(data: SysIdData, order: Union[List[int], Tuple[int, int, int]]) -> LTIModel:
    """
    Estimate ARX model.

    Args:
        data: System identification data.
        order: [na, nb, nk] for SISO.

    Returns:
        LTIModel: Estimated ARX model.
    """
    y_names, u_names = _get_names(data)
    return sysid(data, y_names, u_names, tuple(order), method="arx")


def n4sid(data: SysIdData, order: int) -> LTIModel:
    """
    Estimate State-Space model using N4SID.

    Args:
        data: System identification data.
        order: Number of states.

    Returns:
        LTIModel: Estimated state-space model.
    """
    y_names, u_names = _get_names(data)
    return sysid(data, y_names, u_names, order, method="n4sid")


def oe(data: SysIdData, order: Union[List[int], Tuple[int, int, int]]) -> LTIModel:
    """
    Estimate Output-Error model.

    Args:
        data: System identification data.
        order: [nb, nf, nk].

    Returns:
        LTIModel: Estimated OE model.
    """
    nb, nf, nk = order
    y_names, u_names = _get_names(data)
    # Map to ARX structure (na, nb, nk) where na=nf
    # Note: OE is effectively ARX with specific noise structure, but here we map arguments
    # to the underlying 'oe' method which handles it.
    return sysid(data, y_names, u_names, (nf, nb, nk), method="oe")


def pem(data: SysIdData, order: Union[int, List[int], Tuple[int, ...], None] = None) -> LTIModel:
    """
    Estimate model using Prediction Error Method.

    Args:
        data: System identification data.
        order: If int, estimates state-space model of that order.
               If list/tuple, estimates polynomial model (e.g., ARX/OE structure).

    Returns:
        LTIModel: Estimated model.
    """
    y_names, u_names = _get_names(data)
    if isinstance(order, int):
        # State space PEM
        return sysid(data, y_names, u_names, order, method="pem", settings={"init": "n4sid"})
    else:
        # Polynomial PEM (default init is arx)
        if order is None:
            raise ValueError("Order must be specified for PEM.")
        return sysid(data, y_names, u_names, tuple(order), method="pem")


def compare(data: SysIdData, model: LTIModel) -> float:
    """
    Compare measured output with simulated output.

    Args:
        data: Validation data.
        model: Estimated model.

    Returns:
        float: Fit percentage (0-100) or ratio (0-1).
    """
    y_names, u_names = _get_names(data)

    u_list = [data[name] for name in u_names]
    u = np.column_stack(u_list)

    y_list = [data[name] for name in y_names]
    y = np.column_stack(y_list)

    y_hat = model.simulate(u)
    # model.compare returns 1 - NRMSE (ratio)
    fit_ratio = model.compare(y, u)

    t = np.arange(y.shape[0]) * data.Ts

    plt.figure()
    if y.shape[1] == 1:
        plt.plot(t, y, "k", label="Measured")
        plt.plot(t, y_hat, "b", label=f"Simulated (fit: {fit_ratio:.2%})")
        plt.legend()
        plt.title("Model Comparison")
    else:
        ny = y.shape[1]
        for i in range(ny):
            plt.subplot(ny, 1, i + 1)
            plt.plot(t, y[:, i], "k", label="Measured")
            plt.plot(t, y_hat[:, i], "b", label="Simulated")
            plt.ylabel(y_names[i])
            if i == 0:
                plt.legend()
                plt.title("Model Comparison")

    plt.xlabel("Time [s]")
    plt.show()

    return fit_ratio


def step(model: LTIModel, Tfinal: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot step response.

    Args:
        model: LTI model.
        Tfinal: Final simulation time.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Output response y, time vector t.
    """
    N = 100
    if Tfinal is not None:
        N = int(Tfinal / model.Ts) + 1

    t, y = model.step_response(N=N)

    plt.figure()
    plt.plot(t, y)
    plt.title("Step Response")
    plt.xlabel("Time [s]")
    plt.grid(True, alpha=0.3)
    plt.show()
    return y, t


def impulse(model: LTIModel, Tfinal: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot impulse response.

    Args:
        model: LTI model.
        Tfinal: Final simulation time.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Output response y, time vector t.
    """
    N = 100
    if Tfinal is not None:
        N = int(Tfinal / model.Ts) + 1

    t, y = model.impulse_response(N=N)
    plt.figure()
    plt.stem(t, y)
    plt.title("Impulse Response")
    plt.xlabel("Time [s]")
    plt.grid(True, alpha=0.3)
    plt.show()
    return y, t
