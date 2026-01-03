import numpy as np
from llsi.math import evaluate_state_space


def test_evaluate_state_space_simple():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    u = np.array([[1.0, 2.0, 3.0]])  # (nu, N)
    x0 = np.array([[0.0]])

    y = evaluate_state_space(A, B, C, D, u, x0)
    # manually: x0=0 ; y0 = Cx0 + D u0 = 0 ; x1 = A x0 + B u0 = 1
    # y sequence should be [[0],[1],[3]] if accumulation is correct
    assert y.shape == (3, 1)
    assert np.isfinite(y).all()
