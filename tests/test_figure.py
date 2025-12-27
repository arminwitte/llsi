#!/usr/bin/env python3
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import matplotlib.pyplot as plt

from llsi.figure import Figure


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
