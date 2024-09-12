import numpy as np
from numba import njit


@njit
def evaluate_state_space(A, B, C, D, u, x1):
    N = u.shape[1]
    ny = C.shape[0]
    nu = B.shape[1]
    y = np.empty((N, ny))
    for i, u_ in enumerate(u.T):
        ui = np.ascontiguousarray(u_.T)
        ui = ui.reshape(nu, 1)
        x = x1
        # with np.errstate(over="ignore", invalid="ignore"):
        x1 = A @ x + B @ ui
        y_ = C @ x + D @ ui

        y[i, :] = y_.ravel()
    return y
