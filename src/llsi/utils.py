"""
Utility functions for system identification.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize

from .ltimodel import LTIModel
from .sysiddata import SysIdData


def cv(
    training_data: SysIdData,
    validation_data: SysIdData,
    y_name: Union[str, List[str]],
    u_name: Union[str, List[str]],
    order: Any,
    method: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    bounds: Tuple[float, float] = (0, 100),
) -> Tuple[float, float]:
    """
    Cross-validation to find optimal regularization parameter lambda.

    Args:
        training_data: Data for training.
        validation_data: Data for validation.
        y_name: Output channel name(s).
        u_name: Input channel name(s).
        order: Model order.
        method: Identification method name.
        settings: Base settings dictionary.
        bounds: Bounds for lambda (min, max).

    Returns:
        Tuple[float, float]: Optimal lambda and the corresponding fit score.
    """
    # Import locally to avoid circular dependencies if any
    from .sysidalg import sysid

    if settings is None:
        settings = {}

    def fun(lmb_exp: float) -> float:
        s = settings.copy()
        s["lambda"] = 10**lmb_exp
        mod = sysid(training_data, y_name, u_name, order, method=method, settings=s)
        y = validation_data[y_name]
        u = validation_data[u_name]
        fit = mod.compare(y, u)
        # print(f"lambda={10**lmb_exp:.4e}, fit={fit:.4f}")
        return -fit

    # Ensure bounds are positive for log
    lower_bound = max(bounds[0], 1e-12)
    upper_bound = max(bounds[1], 1e-12)

    bounds_log = (np.log10(lower_bound), np.log10(upper_bound))

    res = scipy.optimize.minimize_scalar(fun, bounds=bounds_log, method="bounded")

    return 10**res.x, -res.fun


def rise_time(mod: LTIModel, N: int = 200) -> float:
    """
    Calculate the rise time (10% to 90%) of the step response.

    Args:
        mod: The LTI model.
        N: Number of simulation steps.

    Returns:
        float: The rise time in seconds.
    """
    t, s = mod.step_response(N=N)

    # Normalize step response if final value is not 1?
    # Usually rise time is defined relative to steady state value.
    # Assuming steady state is reached and is non-zero.
    final_val = s[-1, 0] if s.ndim > 1 else s[-1]
    if np.abs(final_val) < 1e-6:
        return 0.0  # Or raise error

    s_norm = s / final_val

    # Find indices
    # argwhere returns indices where condition is true
    idx_10 = np.argwhere(s_norm > 0.1)
    idx_90 = np.argwhere(s_norm > 0.9)

    if len(idx_10) == 0 or len(idx_90) == 0:
        return float("nan")

    t_10 = t[idx_10[0][0]]
    t_90 = t[idx_90[0][0]]

    return float(t_90 - t_10)


def settling_time(mod: LTIModel, margin: float = 0.01, N: int = 200) -> float:
    """
    Calculate the settling time.

    Args:
        mod: The LTI model.
        margin: The error margin (default 1%).
        N: Number of simulation steps.

    Returns:
        float: The settling time in seconds.
    """
    t, s = mod.step_response(N=N)

    final_val = s[-1, 0] if s.ndim > 1 else s[-1]
    if np.abs(final_val) < 1e-6:
        return 0.0

    s_norm = s / final_val

    upper = 1.0 + margin
    lower = 1.0 - margin

    # Find last time it was outside the margin
    outside_indices = np.argwhere((s_norm > upper) | (s_norm < lower))

    if len(outside_indices) == 0:
        return 0.0  # Always inside?

    last_idx = outside_indices[-1][0]

    # Settling time is the time of the last sample outside the margin?
    # Or the next sample? Usually the time after which it stays inside.
    # So t[last_idx] is the last time it was bad. t[last_idx+1] is good.
    # Let's return t[last_idx] as approximation.

    return float(t[last_idx])
