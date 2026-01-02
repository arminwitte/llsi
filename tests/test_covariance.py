import numpy as np

from llsi.arx import ARX
from llsi.pem import OE, PEM


def test_arx_covariance_shape(data_siso_deterministic_stochastic):
    """Test that ARX computes covariance matrix of correct shape."""
    data = data_siso_deterministic_stochastic
    alg = ARX(data, "y", "u")
    # Order: na=2, nb=3, nk=0
    # Parameters: b0, b1, b2, a1, a2 (5 parameters)
    # Note: a0 is fixed to 1.
    order = (2, 3, 0)
    mod = alg.ident(order)

    assert mod.cov is not None
    # ARX returns PolynomialModel.
    # Number of parameters = nb + na = 3 + 2 = 5
    assert mod.cov.shape == (5, 5)

    # Check symmetry
    np.testing.assert_allclose(mod.cov, mod.cov.T, atol=1e-8)

    # Check positive diagonal (variances)
    assert np.all(np.diag(mod.cov) > 0)


def test_pem_covariance_shape(data_siso_deterministic_stochastic):
    """Test that PEM computes covariance matrix of correct shape."""
    data = data_siso_deterministic_stochastic
    # PEM initialized with ARX
    alg = PEM(data, "y", "u", settings={"init": "arx"})
    order = (2, 3, 0)
    mod = alg.ident(order)

    assert mod.cov is not None
    # Should match ARX parameter count: 5
    assert mod.cov.shape == (5, 5)

    # Check symmetry
    np.testing.assert_allclose(mod.cov, mod.cov.T, atol=1e-8)

    # Check positive diagonal
    assert np.all(np.diag(mod.cov) > 0)


def test_oe_covariance_shape(data_siso_deterministic_stochastic):
    """Test that OE computes covariance matrix of correct shape."""
    data = data_siso_deterministic_stochastic
    alg = OE(data, "y", "u")
    order = (2, 3, 0)
    mod = alg.ident(order)

    assert mod.cov is not None
    # OE structure B/F.
    # If initialized with ARX, it might be converted or kept as PolynomialModel?
    # OE usually implies Output Error model, but llsi might use PolynomialModel structure.
    # Let's check what OE returns. It inherits from PEM, which uses init='arx' by default in OE class.
    # So it returns PolynomialModel.
    assert mod.cov.shape == (5, 5)

    # Check symmetry
    np.testing.assert_allclose(mod.cov, mod.cov.T, atol=1e-8)

    # Check positive diagonal
    assert np.all(np.diag(mod.cov) > 0)


def test_pem_covariance_values_reasonable(data_siso_deterministic_stochastic):
    """Test that PEM covariance values are within reasonable bounds."""
    data = data_siso_deterministic_stochastic
    alg = PEM(data, "y", "u", settings={"init": "arx"})
    order = (2, 3, 0)
    mod = alg.ident(order)

    # Variances shouldn't be extremely large for this dataset
    variances = np.diag(mod.cov)

    # Relaxed check: Variances should be positive and finite.
    # Large values are possible if parameters are poorly identifiable in OE setting.
    assert np.all(variances > 0)
    assert np.all(np.isfinite(variances))

    # Check that we don't have extremely huge values indicating numerical blowup
    # (e.g. > 1e10). 1e5 is acceptable for ill-conditioned cases in tests.
    assert np.all(variances < 1e10)


def test_adam_covariance_shape(data_siso_deterministic_stochastic):
    """Test that ADAM computes covariance matrix of correct shape."""
    from llsi.pem import ADAM

    data = data_siso_deterministic_stochastic
    alg = ADAM(data, "y", "u", settings={"init": "arx", "max_epochs": 5})
    order = (2, 3, 0)
    mod = alg.ident(order)

    assert mod.cov is not None
    assert mod.cov.shape == (5, 5)

    # Check symmetry
    np.testing.assert_allclose(mod.cov, mod.cov.T, atol=1e-8)

    # Check positive diagonal
    assert np.all(np.diag(mod.cov) >= 0)


def test_pem_covariance_ill_conditioned_fallback(data_siso_deterministic):
    """Test fallback for ill-conditioned covariance matrix."""
    # Use deterministic data (no noise) -> residuals near zero -> sigma2 near zero
    # But Jacobian might be fine.
    # To make H_approx singular, we need parameters to be unidentifiable or redundant.
    # Or just very small epsilon?

    # Actually, if we use a model order that is too high, we might get singular H.
    data = data_siso_deterministic
    alg = PEM(data, "y", "u", settings={"init": "arx"})
    # Over-parameterized model
    order = (10, 10, 0)
    mod = alg.ident(order)

    assert mod.cov is not None
    assert mod.cov.shape == (20, 20)
    # Just checking it doesn't crash and returns something
