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
    Perform cross-validation to find the optimal regularization parameter (lambda).

    This function optimizes the `lambda` setting for a system identification method
    by minimizing the negative fit score on a validation dataset.

    Args:
        training_data: The dataset used for training the model.
        validation_data: The dataset used for validating the model.
        y_name: The name(s) of the output channel(s).
        u_name: The name(s) of the input channel(s).
        order: The order of the model to identify.
        method: The identification method to use (e.g., 'n4sid', 'arx').
        settings: A dictionary of base settings for the identification method.
        bounds: A tuple (min, max) specifying the search range for lambda.

    Returns:
        Tuple[float, float]: A tuple containing:
            - The optimal lambda value.
            - The corresponding fit score (higher is better).
    """
    # Import locally to avoid circular dependencies if any
    from .sysidalg import sysid

    if settings is None:
        settings = {}

    def fun(lmb_exp: float) -> float:
        s = settings.copy()
        s["lambda"] = 10**lmb_exp
        mod = sysid(training_data, y_name, u_name, order, method=method, settings=s)

        # Check that required series exist in validation data
        y_names_list = [y_name] if isinstance(y_name, str) else y_name
        u_names_list = [u_name] if isinstance(u_name, str) else u_name
        missing_outputs = [name for name in y_names_list if name not in validation_data]
        missing_inputs = [name for name in u_names_list if name not in validation_data]
        if missing_outputs:
            raise ValueError(f"Output series not found in validation data: {missing_outputs}")
        if missing_inputs:
            raise ValueError(f"Input series not found in validation data: {missing_inputs}")

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


def save_model(model: LTIModel, filename: str) -> None:
    """
    Save an LTI model to a JSON file.

    Args:
        model: The model to save.
        filename: The path to the file.
    """
    if hasattr(model, "to_json"):
        # Add type info to the JSON so we know which class to load
        import json

        # Get the JSON string from the model
        json_str = model.to_json()
        data = json.loads(json_str)

        # Add class name
        data["__class__"] = model.__class__.__name__

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise NotImplementedError(f"Model type {type(model)} does not support to_json serialization.")


def load_model(filename: str) -> LTIModel:
    """
    Load an LTI model from a JSON file.

    Args:
        filename: The path to the file.

    Returns:
        LTIModel: The loaded model.
    """
    import json

    from .polynomialmodel import PolynomialModel
    from .statespacemodel import StateSpaceModel

    with open(filename) as f:
        data = json.load(f)

    class_name = data.get("__class__")

    if class_name == "StateSpaceModel":
        return StateSpaceModel.from_json(filename)
    elif class_name == "PolynomialModel":
        return PolynomialModel.from_json(filename)
    else:
        # Fallback: try to infer from keys
        if "A" in data and "B" in data:
            return StateSpaceModel.from_json(filename)
        elif "a" in data and "b" in data:
            return PolynomialModel.from_json(filename)
        else:
            raise ValueError(f"Unknown model type in {filename}")
