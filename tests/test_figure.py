#!/usr/bin/env python3
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import matplotlib.pyplot as plt
import numpy as np

from llsi.figure import Figure
from llsi.sysiddata import SysIdData


def test_ss(ss_mod):
    with Figure() as fig:
        fig.plot([ss_mod])

    assert isinstance(fig.fig, plt.Figure)


def test_ss_impulse(ss_mod):
    with Figure() as fig:
        fig.plot([ss_mod], "impulse")

    assert isinstance(fig.fig, plt.Figure)


def test_ss_step(ss_mod):
    with Figure() as fig:
        fig.plot([ss_mod], "step")

    assert isinstance(fig.fig, plt.Figure)


def test_ss_frequency(ss_mod):
    with Figure() as fig:
        fig.plot([ss_mod], "frequency")

    assert isinstance(fig.fig, plt.Figure)


# def test_ss_hsv(ss_mod):
#     with Figure() as fig:
#         fig.plot([ss_mod],'hsv')

#     assert isinstance(fig.fig,plt.Figure)


def test_data(data_siso_deterministic_stochastic):
    with Figure() as fig:
        fig.plot([data_siso_deterministic_stochastic])

    assert isinstance(fig.fig, plt.Figure)


def test_residuals(poly_mod):
    # Create dummy data
    u = np.random.randn(100, 1)
    y = poly_mod.simulate(u)
    y_noisy = y + 0.1 * np.random.randn(100, 1)

    data = SysIdData(Ts=1.0, u=u, y=y_noisy)

    with Figure() as fig:
        fig.plot({"mod": poly_mod, "data": data}, "residuals")

    assert isinstance(fig.fig, plt.Figure)
