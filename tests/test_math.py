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
