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
