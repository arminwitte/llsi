#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:24:49 2022

@author: armin
"""

import numpy as np
import pytest
import scipy.stats

from llsi import PolynomialModel, StateSpaceModel, SysIdData


def generate_data(filt, noise=0.01):
    t, u = SysIdData.generate_prbs(10000, 1.0)

    # generate seed
    gen = np.random.Generator(np.random.PCG64(42))
    norm = scipy.stats.norm
    norm.random_state = gen

    e = noise * norm.rvs(size=10000)

    y = filt.simulate(u) + e

    data = SysIdData(y=y, u=u, t=t)
    data.equidistant()
    return data


# @pytest.fixture
# def ss_mod():
#     return StateSpaceModel(A=[[0.8, 0.8], [0, 0.8]], B=[1, 1], C=[1, 0], D=0, Ts=1)


# @pytest.fixture
# def poly_mod(ss_mod):
#     tf = ss_mod.to_tf()
#     return PolynomialModel(a=tf.den, b=tf.num, Ts=1)


@pytest.fixture
def poly_mod():
    return PolynomialModel(
        a=np.array([1.0, -1.66, 0.83]), b=np.array([1.0, 2.0, 1.0]), Ts=1
    )


@pytest.fixture
def ss_mod(poly_mod):
    tf = poly_mod.to_tf()
    ss = tf.to_ss()
    return StateSpaceModel(A=ss.A, B=ss.B, C=ss.C, D=ss.D, Ts=1)


@pytest.fixture
def data_siso_deterministic(ss_mod):
    filt = ss_mod
    data = generate_data(filt, noise=1e-9)
    data.center()
    return data


@pytest.fixture
def data_siso_deterministic_stochastic(ss_mod):
    filt = ss_mod
    data = generate_data(filt, noise=0.5)
    data.center()
    return data
