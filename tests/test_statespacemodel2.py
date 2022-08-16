#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 00:01:33 2022

@author: armin
"""

import numpy as np
import pytest

from llsi.ltimodel import LTIModel
from llsi.statespacemodel import StateSpaceModel


@pytest.fixture
def model():
    return StateSpaceModel(A=[[0.8, 0.8], [0, 0.8]], B=[1, 1], C=[1, 0], D=0, Ts=1)


def test_init():
    ss = StateSpaceModel(A=[[0.8, 0.8], [0, 0.8]], B=[1, 1], C=[1, 0], D=0, Ts=1)
    assert isinstance(ss, LTIModel)
    assert isinstance(ss, StateSpaceModel)
    assert ss.Ts == 1.0


def test_vectorize(model):
    theta = model.vectorize()
    np.testing.assert_allclose(theta, [0.8, 0.8, 0.0, 0.8, 1.0, 1.0, 1.0, 0.0, 0.0])


def test_reshape(model):
    model.vectorize()
    model.reshape(np.array([0.8, 0.7, 0.0, 0.8, 1.0, 1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(model.A, [[0.8, 0.7], [0.0, 0.8]])


def test_simulate(model):
    u = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = model.simulate(u)
    np.testing.assert_allclose(
        y, [0.0, 1.0, 1.6, 1.92, 2.048, 2.048, 1.96608, 1.835008, 1.677722], rtol=1e-6
    )


def test_from_PT1():
    mod = StateSpaceModel.from_PT1(1.0, 0.8, Ts=1.0)
    np.testing.assert_allclose(mod.A, [[0.2307691]], rtol=1e-6)


def test_repr_str(model):
    print(model.__repr__())
    s = """A:
[[0.8 0.8]
 [0.  0.8]]
B:
[[1]
 [1]]
C:
[[1 0]]
D:
0
"""
    assert model.__repr__() == s
    assert model.__str__() == s


def test_impulse(model):
    ti, i = model.impulse_response()
    np.testing.assert_allclose(
        i[:10],
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
        I[:10],
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
