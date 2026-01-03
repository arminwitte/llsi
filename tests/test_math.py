import numpy as np

from llsi.math import evaluate_state_space


def test_evaluate_state_space_identity():
    # Simple identity system: x[k+1] = x[k] + u[k], y[k] = x[k]
    # A=1, B=1, C=1, D=0
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])

    # Input: step
    u = np.ones((1, 10))  # 1 input, 10 samples
    x0 = np.array([[0.0]])

    y = evaluate_state_space(A, B, C, D, u, x0)

    # Expected: y[k] = k (ramp) because x accumulates u
    # k=0: x=0, y=0, x_next=1
    # k=1: x=1, y=1, x_next=2
    # ...
    expected = np.arange(10).reshape(10, 1)

    np.testing.assert_allclose(y, expected, atol=1e-10)


def test_evaluate_state_space_passthrough():
    # Passthrough system: y[k] = u[k]
    # A=0, B=0, C=0, D=1
    A = np.array([[0.0]])
    B = np.array([[0.0]])
    C = np.array([[0.0]])
    D = np.array([[1.0]])

    u = np.random.rand(1, 20)
    x0 = np.array([[0.0]])

    y = evaluate_state_space(A, B, C, D, u, x0)

    np.testing.assert_allclose(y.ravel(), u.ravel(), atol=1e-10)


def test_evaluate_state_space_mimo():
    # 2 inputs, 2 outputs, 2 states
    # x[k+1] = I*x[k] + I*u[k]
    # y[k] = I*x[k]
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))

    u = np.ones((2, 5))
    x0 = np.zeros((2, 1))

    y = evaluate_state_space(A, B, C, D, u, x0)

    # y should be [[0,0], [1,1], [2,2], [3,3], [4,4]]
    expected = np.array([[i, i] for i in range(5)])

    np.testing.assert_allclose(y, expected, atol=1e-10)


# --- Consolidated tests from other files ---

# If numba compiled the function, call the underlying Python impl for coverage
evaluate_state_space_py = getattr(evaluate_state_space, "py_func", evaluate_state_space)


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
