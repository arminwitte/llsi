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
from llsi.statespacemodel import StateSpaceModel


@pytest.fixture
def model():
    return StateSpaceModel(A=[[0.8, 0.8], [0, 0.8]], B=[1, 1], C=[1, 0], D=0, Ts=1)


def test_init():
    ss = StateSpaceModel()
    assert isinstance(ss, LTIModel)
    assert isinstance(ss, StateSpaceModel)
    assert ss.Ts == 1.0


def test_init_siso():
    ss = StateSpaceModel(A=[[0.8, 0.8], [0, 0.8]], B=[1, 1], C=[1, 0], D=0, Ts=1.0)
    assert isinstance(ss, LTIModel)
    assert isinstance(ss, StateSpaceModel)
    assert ss.D == np.array([[0.0]])
    assert ss.Ts == 1.0


def test_init_mimo():
    ss = StateSpaceModel(
        A=[[0.8, 0.8], [0, 0.8]],
        B=[[1, 1], [1, 1]],
        C=[[1, 0], [0, 1]],
        D=[[0, 0], [0, 0]],
        Ts=1.0,
    )
    assert isinstance(ss, LTIModel)
    assert isinstance(ss, StateSpaceModel)
    assert ss.Ts == 1.0


def test_vectorize(ss_mod):
    theta = ss_mod.vectorize()
    print(theta)
    np.testing.assert_allclose(
        theta, [1.66, -0.83, 1.0, 0.0, 1.0, 0.0, 3.66, 0.17, 1.0, 0.0, 0.0]
    )


def test_reshape(model):
    model.vectorize()
    model.reshape(np.array([0.8, 0.7, 0.0, 0.8, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(model.A, [[0.8, 0.7], [0.0, 0.8]])


def test_simulate(model):
    u = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = model.simulate(u)
    assert y.shape == (9, 1)
    np.testing.assert_allclose(
        y.ravel(),
        [0.0, 1.0, 1.6, 1.92, 2.048, 2.048, 1.96608, 1.835008, 1.677722],
        rtol=1e-6,
    )


def test_from_PT1():
    mod = StateSpaceModel.from_PT1(1.0, 0.8, Ts=1.0)
    np.testing.assert_allclose(mod.A, [[0.2307691]], rtol=1e-6)


def test_repr_str(model):
    s = """StateSpaceModel with Ts=1
A:
[[0.8 0.8]
 [0.  0.8]]
B:
[[1]
 [1]]
C:
[[1 0]]
D:
[[0]]
"""
    assert model.__repr__() == s
    assert model.__str__() == s


def test_impulse(model):
    ti, i = model.impulse_response()
    np.testing.assert_allclose(
        i.ravel()[:10],
        [
            0.00000000e00,
            1.00000000e00,
            1.60000000e00,
            1.92000000e00,
            2.04800000e00,
            2.04800000e00,
            1.96608000e00,
            1.83500800e00,
            1.67772160e00,
            1.50994944e00,
        ],
    )


def test_step(model):
    tI, I = model.step_response()
    np.testing.assert_allclose(
        I.ravel()[:10],
        [
            0.0,
            1.0,
            2.6,
            4.52,
            6.568,
            8.616,
            10.58208,
            12.417088,
            14.0948096,
            15.60475904,
        ],
    )


def test_to_ss(ss_mod):
    ss = ss_mod.to_ss()
    assert isinstance(ss, scipy.signal.StateSpace)
    tf = ss.to_tf()
    ss = tf.to_ss()
    print(ss.A)
    np.testing.assert_allclose(ss.A, [[1.66, -0.83], [1.0, 0.0]])


def test_to_ss_continuous(ss_mod):
    ss = ss_mod.to_ss(continuous=True)
    assert isinstance(ss, scipy.signal.StateSpace)

    print(ss.A)
    np.testing.assert_allclose(
        ss.A, [[0.85386819, -0.9512894], [1.14613181, -1.0487106]]
    )


#     tf = ss.to_tf()
#     ss = tf.to_ss()
#     np.testing.assert_allclose(ss.A, [[-0.19484241, -0.19484241],
#  [ 1.,          0.        ]]
# )


def test_to_ss_continuous_euler(ss_mod):
    ss = ss_mod.to_ss(continuous=True, method="euler")
    assert isinstance(ss, scipy.signal.StateSpace)
    print(ss.A)
    np.testing.assert_allclose(ss.A, [[0.66, -0.83], [1.0, -1.0]])

    tf = ss.to_tf()
    ss = tf.to_ss()
    np.testing.assert_allclose(ss.A, [[-0.34, -0.17], [1.0, 0.0]])


def test_to_tf(ss_mod):
    tf = ss_mod.to_tf()
    assert isinstance(tf, scipy.signal.TransferFunction)
    np.testing.assert_allclose(tf.den, [1.0, -1.66, 0.83])


def test_to_zpk(ss_mod):
    zpk = ss_mod.to_zpk()
    assert isinstance(zpk, scipy.signal.ZerosPolesGain)
    print(zpk.poles)
    np.testing.assert_allclose(zpk.poles, [0.83 + 0.3756328j, 0.83 - 0.3756328j])


def test_to_controllable_form(ss_mod):
    ss = ss_mod.to_controllable_form()
    assert isinstance(ss, StateSpaceModel)
    np.testing.assert_allclose(ss.A, [[1.66, -0.83], [1.0, 0.0]])


def test_reduce_order(ss_mod):
    red_mod, s = ss_mod.reduce_order(1)
    print(red_mod.A)
    np.testing.assert_allclose(red_mod.A, [[0.84831905]])
    print(s)
    np.testing.assert_allclose(s, [33.17635127, 21.41164539])
