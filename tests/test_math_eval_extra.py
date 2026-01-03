import numpy as np
from llsi.math import evaluate_state_space

# If numba compiled the function, call the underlying Python impl for coverage
evaluate_state_space_py = getattr(evaluate_state_space, "py_func", evaluate_state_space)


def test_evaluate_state_space_single_step():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    u = np.array([[2.0]])  # (nu=1, N=1)
    x0 = np.array([[0.0]])
    y = evaluate_state_space_py(A, B, C, D, u, x0)
    assert y.shape == (1, 1)
    assert np.allclose(y, 0.0)  # y[0] = C x0 + D u = 0


def test_evaluate_state_space_multi_step():
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    u = np.array([[1.0, 2.0, 3.0]])  # (nu=1, N=3)
    x0 = np.array([[0.0]])
    y = evaluate_state_space_py(A, B, C, D, u, x0)
    assert y.shape == (3, 1)
    # With A=0, state equals B*u[k-1], but first output equals 0 (x0)
    assert np.allclose(y[0, 0], 0.0)
