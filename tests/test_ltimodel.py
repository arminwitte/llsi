#!/usr/bin/env python3
"""
Created on Sun Aug  7 00:01:33 2022

@author: armin
"""

import numpy as np
import pytest

from llsi.ltimodel import LTIModel
from llsi.sysidalgbase import compute_residuals_analysis
from llsi.sysiddata import SysIdData


@pytest.fixture
def e():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_hat = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9])
    return LTIModel.residuals(y, y_hat)


def test_NRMSE():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_hat = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9])
    np.testing.assert_allclose(LTIModel.NRMSE(y, y_hat), 0.03481553119113951)


def test_residuals():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_hat = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9])
    np.testing.assert_allclose(
        LTIModel.residuals(y, y_hat),
        [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
    )


def test_SE(e):
    np.testing.assert_allclose(LTIModel.SE(e), [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])


def test_SSE(e):
    np.testing.assert_allclose(LTIModel.SSE(e), [0.1])


def test_MSE(e):
    print(LTIModel.MSE(e))
    np.testing.assert_allclose(LTIModel.MSE(e), [0.01])


def test_RMSE(e):
    print(LTIModel.RMSE(e))
    np.testing.assert_allclose(LTIModel.RMSE(e), [0.1])


def test_compute_residuals_analysis(poly_mod):
    # Create dummy data
    u = np.random.randn(100, 1)
    y = poly_mod.simulate(u)

    # Add some noise to make residuals non-zero
    y_noisy = y + 0.1 * np.random.randn(100, 1)

    data = SysIdData(Ts=1.0, u=u, y=y_noisy)

    res_analysis = compute_residuals_analysis(poly_mod, data)

    assert "residuals" in res_analysis
    assert "acf" in res_analysis
    assert "ccf" in res_analysis
    assert "lags" in res_analysis
    assert "conf_interval" in res_analysis

    assert len(res_analysis["acf"]) == len(res_analysis["lags"])
    assert len(res_analysis["ccf"]) == len(res_analysis["lags"])
    assert isinstance(res_analysis["conf_interval"], float)
