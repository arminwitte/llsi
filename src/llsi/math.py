"""
Mathematical utility functions, primarily for accelerated simulation.
"""

import numpy as np

try:
    from numba import njit
except Exception:
    # Fallback if numba is not installed or import fails.
    # Support both @njit and @njit(...kwargs...) usage.
    def njit(*args, **kwargs):
        # Used as @njit without args
        if args and callable(args[0]) and not kwargs:
            return args[0]

        # Used as @njit(...) with args/kwargs -> return decorator
        def _decorator(func):
            return func

        return _decorator


@njit
def evaluate_state_space(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray,
) -> np.ndarray:
    """
    Simulate a discrete-time state-space system using Numba.

    x[k+1] = A x[k] + B u[k]
    y[k]   = C x[k] + D u[k]

    Args:
        A: State transition matrix (nx, nx).
        B: Input matrix (nx, nu).
        C: Output matrix (ny, nx).
        D: Feedthrough matrix (ny, nu).
        u: Input signal array of shape (nu, N).
        x0: Initial state vector of shape (nx, 1).

    Returns:
        np.ndarray: Output signal array of shape (N, ny).
    """
    N = u.shape[1]
    ny = C.shape[0]
    nu = B.shape[1]

    y = np.empty((N, ny))

    # Initialize current state
    x = x0

    # Iterate over time steps
    # u.T is (N, nu)
    u_T = u.T

    for i in range(N):
        # Extract input vector for current time step
        ui = np.ascontiguousarray(u_T[i]).reshape(nu, 1)

        # Calculate output: y[k] = C x[k] + D u[k]
        y_ = C @ x + D @ ui
        y[i, :] = y_.ravel()

        # Update state: x[k+1] = A x[k] + B u[k]
        x = A @ x + B @ ui

    return y


@njit
def prbs31(code: int) -> int:
    """
    Single step of PRBS31 generator.
    Polynomial: x^31 + x^28 + 1
    """
    feedback = ((code >> 30) ^ (code >> 27)) & 1
    return ((code << 1) | feedback) & 0x7FFFFFFF


@njit
def generate_prbs_sequence(N: int, seed: int) -> np.ndarray:
    """
    Generate PRBS31 sequence of length N.
    Returns array with values 0.0 and 1.0.
    """
    u = np.empty(N, dtype=np.float64)
    state = int(seed) & 0x7FFFFFFF
    if state == 0:
        state = 1

    for i in range(N):
        u[i] = float(state & 1)
        state = prbs31(state)

    return u


# =============================================================================
# Output Error (OE) Model Acceleration Functions
# =============================================================================


@njit
def oe_simulate(u: np.ndarray, b: np.ndarray, f: np.ndarray, nk: int) -> np.ndarray:
    """
    Simulate OE model: y[k] = (B/F) * u[k-nk]

    This is a Numba-optimized replacement for scipy.signal.lfilter for OE models.
    The model structure is: y[k] = -f1*y[k-1] - f2*y[k-2] - ... + b0*u[k-nk] + b1*u[k-nk-1] + ...

    Args:
        u: Input signal (N,)
        b: Numerator coefficients [b0, b1, ...] (nb,)
        f: Denominator coefficients [1.0, f1, f2, ...] (nf,). f[0] must be 1.0.
        nk: Input delay (samples)

    Returns:
        y: Simulated output (N,)
    """
    N = len(u)
    nf = len(f)
    nb = len(b)

    if N == 0 or nf == 0 or nb == 0:
        return np.zeros(N)

    y = np.zeros(N)

    for k in range(N):
        # B part (input contribution)
        val = 0.0
        for j in range(nb):
            idx_u = k - nk - j
            if idx_u >= 0:
                val += b[j] * u[idx_u]

        # F part (feedback from output)
        for i in range(1, nf):
            if k - i >= 0:
                val -= f[i] * y[k - i]

        y[k] = val

    return y


@njit
def oe_cost_and_gradient(
    theta: np.ndarray,
    u: np.ndarray,
    y_true: np.ndarray,
    nb: int,
    nf: int,
    nk: int,
) -> tuple[float, np.ndarray]:
    """
    Compute SSE and analytical gradient for OE model in a single pass.

    This is the "Turbo" for OE identification. Instead of using finite differences
    (which requires N_params simulations per optimization step), this computes the
    exact analytical gradient using sensitivity filtering.

    The gradient is computed as:
    - ∂ŷ/∂b_j = u_filt[k-j]  (u filtered by 1/F)
    - ∂ŷ/∂f_i = -y_filt[k-i] (y_sim filtered by 1/F)

    Args:
        theta: Parameter vector [b0, b1, ..., f1, f2, ...] (n_params,)
        u: Input signal (N,)
        y_true: True output signal (N,)
        nb: Number of B coefficients
        nf: Number of F coefficients (including leading 1.0)
        nk: Input delay (samples)

    Returns:
        sse: Sum of squared errors (float)
        grad: Gradient vector (n_params,)
    """
    N = len(u)
    n_params = len(theta)

    if N == 0 or n_params == 0:
        return 0.0, np.zeros(n_params)

    # Unpack parameters
    b = theta[:nb]
    f_coeffs = theta[nb:]
    # Reconstruct F polynomial: [1.0, f1, f2, ...]
    f = np.concatenate((np.array([1.0]), f_coeffs))

    # Initialize arrays
    y_sim = np.zeros(N)
    u_filt = np.zeros(N)  # u filtered by 1/F
    y_filt = np.zeros(N)  # y_sim filtered by 1/F
    grad = np.zeros(n_params)
    sse = 0.0

    # Compute the start index where all regressors are available
    start = max(nf, nb + nk)

    # Initialize filters for k < start
    # For k < start, we need to compute the filters but not accumulate gradients
    for k in range(N):
        # --- 1. Simulation step ---
        # Numerator (B part)
        val_num = 0.0
        for j in range(nb):
            idx_u = k - nk - j
            if idx_u >= 0:
                val_num += b[j] * u[idx_u]

        # Denominator (F part) - feedback
        val_den = 0.0
        for i in range(1, nf):
            if k - i >= 0:
                val_den += f[i] * y_sim[k - i]

        y_sim[k] = val_num - val_den

        # --- 2. Sensitivity filtering (1/F) ---
        # Filter u[k-nk] with 1/F: u_filt[k] = u[k-nk] - f1*u_filt[k-1] - f2*u_filt[k-2] - ...
        curr_u = u[k - nk] if (k - nk) >= 0 else 0.0
        u_filt[k] = curr_u
        for i in range(1, nf):
            if k - i >= 0:
                u_filt[k] -= f[i] * u_filt[k - i]

        # Filter y_sim[k] with 1/F: y_filt[k] = y_sim[k] - f1*y_filt[k-1] - f2*y_filt[k-2] - ...
        y_filt[k] = y_sim[k]
        for i in range(1, nf):
            if k - i >= 0:
                y_filt[k] -= f[i] * y_filt[k - i]

        # --- 3. Gradient accumulation (only for k >= start) ---
        if k >= start:
            # Error
            err = y_true[k] - y_sim[k]
            sse += err * err

            # dJ/dtheta = -2 * err * dy/dtheta

            # Gradient for b_j: dy/db_j = u_filt[k-j]
            for j in range(nb):
                if (k - j) >= 0:
                    grad[j] += -2 * err * u_filt[k - j]

            # Gradient for f_i: dy/df_i = -y_filt[k-i]
            for i in range(1, nf):
                idx_grad = nb + (i - 1)
                if (k - i) >= 0:
                    grad[idx_grad] += -2 * err * (-y_filt[k - i])

    return sse, grad
