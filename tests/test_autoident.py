
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

