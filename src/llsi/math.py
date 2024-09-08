from numba import njit
import numpy as np

@njit
def evaluate_state_space(A, B, C, D, u, x1):
    N = u.shape[1]
    ny = C.shape[0]
    nu = B.shape[1]
    y = np.empty((N, ny))
    for i, u_ in enumerate(u.T):
        u_ = u_.T
        u_ = u_.reshape(nu, 1)
        x = x1
        # with np.errstate(over="ignore", invalid="ignore"):
        x1 = A @ x + B @ u_
        y_ = C @ x + D @ u_

        y[i, :] = y_.ravel()
    return y
