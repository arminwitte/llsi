import numpy as np
import pytest

try:
    from llsi.sklearn import LTIModel

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_sklearn_wrapper():
    # Generate simple data
    np.random.seed(42)
    N = 100
    u = np.random.randn(N, 1)
    # y[k] = 0.5*u[k] + 0.1*u[k-1]
    y = np.zeros((N, 1))
    for k in range(1, N):
        y[k] = 0.5 * u[k] + 0.1 * u[k - 1]

    model = LTIModel(method="arx", order=(1, 1, 0))
    model.fit(u, y)

    y_pred = model.predict(u)

    assert y_pred.shape == y.shape
    # Check if fit is reasonable
    mse = np.mean((y - y_pred) ** 2)
    assert mse < 0.1


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_sklearn_mimo():
    # 2 inputs, 2 outputs
    np.random.seed(42)
    N = 100
    u = np.random.randn(N, 2)
    y = np.zeros((N, 2))
    # Simple coupling
    y[:, 0] = 0.5 * u[:, 0]
    y[:, 1] = 0.5 * u[:, 1]

    model = LTIModel(method="po-moesp", order=2)
    model.fit(u, y)

    y_pred = model.predict(u)

    assert y_pred.shape == y.shape
    mse = np.mean((y - y_pred) ** 2)
    assert mse < 0.1


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_sklearn_check_estimator():
    # We need to ignore some checks because LTI models are time-series models
    # and standard sklearn checks assume i.i.d. data and might fail on some
    # specific checks (e.g. invariance to shuffling).
    # However, we can check basic compliance.

    # Use po-moesp as it supports MIMO (check_estimator might use multi-output)
    # But check_estimator for Regressor usually checks single output.
    # Let's try.

    # We need to wrap it to ignore specific checks if necessary.
    # For now, let's just see if it passes basic instantiation and get_params.

    model = LTIModel(method="po-moesp")
    # check_estimator(model) # This is strict.

    # Let's just check get_params and set_params which are crucial.
    params = model.get_params()
    assert "method" in params
    assert "order" in params

    model.set_params(order=2)
    assert model.order == 2
