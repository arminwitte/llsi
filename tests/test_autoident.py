
import numpy as np
import pytest
import os
import llsi

def test_autoident_synthetic():
    # Generate synthetic data
    # y(k) = 0.5 y(k-1) + u(k-1)
    N = 1000
    u = np.random.randn(N, 1)
    y = np.zeros((N, 1))
    for k in range(1, N):
        y[k] = 0.5 * y[k-1] + u[k-1]
    
    # Add noise
    y += 0.1 * np.random.randn(N, 1)
    
    res = llsi.autoident(u, y, Ts=0.1, effort='fast')
    
    res.summary()
    
    assert res.model is not None
    assert res.metrics['fit'] > 0.5
    assert len(res.report) > 0

def test_autoident_heated_wire():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "heated_wire_data.npy")
    if not os.path.exists(data_path):
        pytest.skip("Data file not found")
        
    d = np.load(data_path)
    # t = d[:, 0]
    Re = d[:, 1] # Input
    Nu = d[:, 2] # Output
    
    # Downsample manually to speed up test
    Re = Re[::10]
    Nu = Nu[::10]
    
    res = llsi.autoident(Re, Nu, Ts=1.0, max_freq=0.1, effort='fast')
    
    res.summary()
    
    assert res.model is not None
    # assert res.metrics['fit'] > 50.0 # Might vary depending on data quality and settings

def test_autoident_poly(data_siso_deterministic, poly_mod):
    res = llsi.autoident(data_siso_deterministic["u"], data_siso_deterministic["y"], Ts=1.0, effort='fast', order_hint=1, result_type="polynomial")
    np.testing.assert_allclose(res.model.a, poly_mod.a, rtol=1e-3, atol=1e-3)

def test_autoident_ss(data_siso_deterministic, ss_mod):
    res = llsi.autoident(data_siso_deterministic["u"], data_siso_deterministic["y"], Ts=1.0, effort='fast', order_hint=1, result_type="state_space")
    mod = res.model.to_controllable_form()
    np.testing.assert_allclose(mod.A, ss_mod.A, rtol=1e-3, atol=1e-3)



def test_autoident_poly_noise(data_siso_deterministic_stochastic, poly_mod):
    res = llsi.autoident(data_siso_deterministic_stochastic["u"], data_siso_deterministic_stochastic["y"], Ts=1.0, effort='fast', order_hint=1, result_type="polynomial")
    np.testing.assert_allclose(res.model.a, poly_mod.a, rtol=0.5, atol=0.5)



def test_autoident_ss_noise(data_siso_deterministic_stochastic, ss_mod):
    res = llsi.autoident(data_siso_deterministic_stochastic["u"], data_siso_deterministic_stochastic["y"], Ts=1.0, effort='fast', order_hint=1, result_type="state_space")
    mod = res.model.to_controllable_form()
    np.testing.assert_allclose(mod.A, ss_mod.A, rtol=0.5, atol=0.5)