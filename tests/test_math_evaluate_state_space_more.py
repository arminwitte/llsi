import numpy as np
from llsi.math import evaluate_state_space

# call Python impl when numba wraps the function
evaluate_state_space_py = getattr(evaluate_state_space, "py_func", evaluate_state_space)


def test_multi_dimensional_state_and_input():
    # nx=2, nu=2, ny=2, N=3
    A = np.array([[1.0, 0.1], [0.0, 0.9]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    C = np.eye(2)
    D = np.zeros((2, 2))
    u = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    x0 = np.array([[0.0], [1.0]])
    y = evaluate_state_space_py(A, B, C, D, u, x0)
    assert y.shape == (3, 2)


def test_nonzero_D_affects_output_immediately():
    A = np.zeros((1, 1))
    B = np.zeros((1, 1))
    C = np.array([[0.0]])
    D = np.array([[2.0]])
    u = np.array([[3.0, 4.0]])
    x0 = np.array([[0.0]])
    y = evaluate_state_space_py(A, B, C, D, u, x0)
    # y should be D * u at each time step
    assert np.allclose(y.ravel(), np.array([6.0, 8.0]))


def test_initial_state_influence_next_step():
    A = np.array([[0.5]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    u = np.array([[0.0, 0.0]])
    x0 = np.array([[2.0]])
    y = evaluate_state_space_py(A, B, C, D, u, x0)
    # first output is C*x0
    assert np.isclose(y[0, 0], 2.0)
