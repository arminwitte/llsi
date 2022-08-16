#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 00:01:33 2022

@author: armin
"""
import numpy as np
import pytest
import scipy.signal

from llsi.ltimodel import LTIModel


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
    np.testing.assert_allclose(
        LTIModel.SE(e), [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    )


def test_SSE(e):
    np.testing.assert_allclose(LTIModel.SSE(e), [0.1])


def test_MSE(e):
    print(LTIModel.MSE(e))
    np.testing.assert_allclose(LTIModel.MSE(e), [0.01])


def test_RMSE(e):
    print(LTIModel.RMSE(e))
    np.testing.assert_allclose(LTIModel.RMSE(e), [0.1])
