#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np
import pytest
import scipy.stats

from llsi import StateSpaceModel, SysIdData, sysid


def generate_data(filt, noise=0.01):
    # t = np.linspace(0, 9999 * filt.Ts, 10000)
    # u = scipy.stats.norm.rvs(size=10000)
    t, u = SysIdData.generate_prbs(10000, 1.0)
    e = noise * scipy.stats.norm.rvs(size=10000)

    y = filt.simulate(u) + e

    data = SysIdData(y=y, u=u, t=t)
    data.equidistant()
    return data


@pytest.fixture
def data_siso_deterministic():
    filt = StateSpaceModel(A=[[0.8, 0.8], [0, 0.8]], B=[1, 1], C=[1, 0], D=0, Ts=1)
    ti, i = filt.impulse_response(plot=False)
    data = generate_data(filt, noise=1e-9)
    data.center()
    return data


@pytest.fixture
def data_siso_deterministic_stochastic():
    filt = StateSpaceModel(A=[[0.8, 0.8], [0, 0.8]], B=[1, 1], C=[1, 0], D=0, Ts=1)
    ti, i = filt.impulse_response(plot=False)
    data = generate_data(filt, noise=0.1)
    data.center()
    return data


def test_n4sid_deterministic(data_siso_deterministic):
    mod = sysid(data_siso_deterministic, "y", "u", (2), method="n4sid")
    # np.testing.assert_allclose(mod.info["Hankel singular values"], [1200., 160., 0., 0., 0.], rtol=1e-2, atol=1e-2)
    ti, i = mod.impulse_response(plot=False)
    # np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-2, atol=1e-2)
    # np.testing.assert_allclose(
    #     mod.to_controllable_form().A, [[1.6, -0.64], [1.0, 0.0]], rtol=1e-3, atol=1e-3
    # )


def test_po_moesp_deterministic(data_siso_deterministic):
    mod = sysid(data_siso_deterministic, "y", "u", (2), method="po-moesp")
    # np.testing.assert_allclose(mod.info["Hankel singular values"], [0.12, 0.016, 0., 0., 0.], rtol=1e-2, atol=1e-2)
    ti, i = mod.impulse_response(plot=False)
    np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        mod.to_controllable_form().A, [[1.6, -0.64], [1.0, 0.0]], rtol=1e-3, atol=1e-3
    )


# def test_pem_ss_deterministic(data_siso_deterministic):
#     mod = sysid(data_siso_deterministic, "y", "u", (2), method="pem_ss")
#     ti, i = mod.impulse_response(plot=False)
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)


def test_arx_deterministic(data_siso_deterministic):
    mod = sysid(data_siso_deterministic, "y", "u", (2, 3, 0), method="arx")
    ti, i = mod.impulse_response(plot=True)
    np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)

def test_fir_deterministic(data_siso_deterministic):
    mod = sysid(data_siso_deterministic, "y", "u", (0, 50, 0), method="arx")
    ti, i = mod.impulse_response(plot=True)
    import matplotlib.pyplot as plt
    plt.show() 
    np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)


def test_n4sid_deterministic_stochastic(data_siso_deterministic_stochastic):
    mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2), method="n4sid")
    ti, i = mod.impulse_response(plot=False)
    np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)


def test_po_moesp_deterministic_stochastic(data_siso_deterministic_stochastic):
    mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2), method="po-moesp")
    ti, i = mod.impulse_response(plot=False)
    # import matplotlib.pyplot as plt
    # plt.show()
    np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)


# def test_pem_ss_deterministic_stochastic(data_siso_deterministic_stochastic):
#     mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2), method="pem_ss")
#     ti, i = mod.impulse_response(plot=False)
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)


def test_arx_deterministic_stochastik(data_siso_deterministic_stochastic):
    mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2, 3, 0), method="arx")
    ti, i = mod.impulse_response(plot=False)
    np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)
