#!/usr/bin/env python3
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
    return StateSpaceModel(
        A=[[0.8, 0.8], [0, 0.8]],
        B=[1, 1],
        C=[1, 0],
        D=0,
        Ts=1,
        input_names=["input"],
        output_names=["output"],
    )


@pytest.fixture
def model_mimo():
    return StateSpaceModel(
        A=[[0.81867495, -0.00333706], [-0.007094, 0.86617556]],
        B=[[-0.00337626, -0.07836226], [0.04858446, -0.01158329]],
        C=[[-0.37475845, 2.53533251], [-4.17568704, -0.29021108]],
        D=[[6.66661097e-02, -4.52249005e-07], [-6.33970438e-07, 1.81817685e-01]],
        Ts=1,
        input_names=["input"],
        output_names=["output"],
    )


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
    np.testing.assert_allclose(theta, [1.66, -0.83, 1.0, 0.0, 1.0, 0.0, 3.66, 0.17, 1.0, 0.0, 0.0])


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
input(s): ['input']
output(s): ['output']
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
    t_step, y_step = model.step_response()
    np.testing.assert_allclose(
        y_step.ravel()[:10],
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


def test_frequency(model):
    omega, H = model.frequency_response()
    print(H)
    np.testing.assert_allclose(
        H[:10],
        [
            [[24.99848755 - 2.24990963e-01j]],
            [[24.99758034 - 2.84574136e-01j]],
            [[24.99612902 - 3.59931207e-01j]],
            [[24.99380729 - 4.55232791e-01j]],
            [[24.99009335 - 5.75746769e-01j]],
            [[24.98415285 - 7.28121382e-01j]],
            [[24.97465213 - 9.20735643e-01j]],
            [[24.95946063 - 1.16412706e00j]],
            [[24.93517764 - 1.47150140e00j]],
            [[24.8963826 - 1.85931464e00j]],
        ],
    )


def test_step_mimo(model_mimo):
    t_step, y_step = model_mimo.step_response()
    print(y_step)
    np.testing.assert_allclose(
        y_step[:10],
        [
            [0.06666566, 0.18181705],
            [0.19110813, 0.51239338],
            [0.2989585, 0.78286518],
            [0.39242902, 1.00416053],
            [0.47343699, 1.18522055],
            [0.54364404, 1.33336075],
            [0.60449029, 1.45456652],
            [0.65722383, 1.55373501],
            [0.70292633, 1.63487298],
            [0.74253525, 1.70125869],
        ],
    )


def test_frequency_mimo(model_mimo):
    omega, H = model_mimo.frequency_response()
    print(H)
    np.testing.assert_allclose(
        H[:3],
        [
            [
                [9.99954954e-01 - 6.99980408e-03j, -6.23822459e-06 + 5.15998986e-08j],
                [3.36301580e-06 - 1.15636248e-07j, 1.99994478e00 - 9.99971819e-03j],
            ],
            [
                [9.99925564e-01 - 8.85347893e-03j, -6.23798087e-06 + 6.52640751e-08j],
                [3.36224771e-06 - 1.46254530e-07j, 1.99991479e00 - 1.26480067e-02j],
            ],
            [
                [9.99878547e-01 - 1.11978433e-02j, -6.23759099e-06 + 8.25448998e-08j],
                [3.36101903e-06 - 1.84973444e-07j, 1.99986681e00 - 1.59975147e-02j],
            ],
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
    np.testing.assert_allclose(ss.A, [[0.85386819, -0.9512894], [1.14613181, -1.0487106]])


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
