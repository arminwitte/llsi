"""
Mathematical utility functions, primarily for accelerated simulation.
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    # Fallback if numba is not installed
    def njit(func):
        return func


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


# PRBS core implementation moved here so numba-decorated functions live in math.py
def _prbs_core(N: int, seed: int) -> np.ndarray:
    state = int(seed) & 0x7FFFFFFF
    if state == 0:
        state = 1
    u = np.empty(N, dtype=np.float64)
    for i in range(N):
        # output MSB-like bit to match previous implementation ordering
        u[i] = float((state >> 0) & 1)
        # taps for PRBS31: x^31 + x^28 + 1 -> taps at bit positions 30 and 27 (0-based)
        feedback = ((state >> 30) ^ (state >> 27)) & 1
        state = ((state << 1) & 0x7FFFFFFF) | feedback
    return u


# If numba is available, JIT compile the PRBS core
try:
    from numba import njit as _njit

    _prbs_core = _njit(_prbs_core)
except Exception:
    # numba not available; keep Python implementation
    pass

# Public alias
prbs_core = _prbs_core


def prbs31(code: int) -> int:
    """PRBS31 generator step."""
    for _ in range(32):
        next_bit = ~((code >> 30) ^ (code >> 27)) & 0x01
        code = ((code << 1) | next_bit) & 0xFFFFFFFF
    return code


def prbs31_fast(code: int) -> int:
    """Fast PRBS31 generator step."""
    next_code = ~((code << 1) ^ (code << 4)) & 0xFFFFFFF0
    next_code |= ~(((code << 1 & 0x0E) | (next_code >> 31 & 0x01)) ^ (next_code >> 28)) & 0x0000000F
    return next_code


# Attempt to JIT compile the small helpers too
try:
    _prbs31 = _njit(prbs31)
    _prbs31_fast = _njit(prbs31_fast)
    prbs31 = _prbs31
    prbs31_fast = _prbs31_fast
except Exception:
    pass
