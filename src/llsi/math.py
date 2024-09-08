from numba import njit

@njit
def evaluate_state_space(A, B, C, D, u, x1):
    N = u.shape[1]
    y = np.empty((N, self.ny))
    for i, u_ in enumerate(u.T):
        u_ = u_.T
        u_ = u_.reshape(self.nu, 1)
        x = x1
        with np.errstate(over="ignore", invalid="ignore"):
            x1 = self.A @ x + self.B @ u_
            y_ = self.C @ x + self.D @ u_

        y[i, :] = y_.ravel()
    return y
