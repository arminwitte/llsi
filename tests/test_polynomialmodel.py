#!/usr/bin/env python3
"""
Created on Sun Aug  7 00:01:33 2022

@author: armin
"""

import numpy as np
import pytest
import scipy.signal

from llsi.ltimodel import LTIModel
from llsi.polynomialmodel import PolynomialModel


@pytest.fixture
def model():
    return PolynomialModel(
        b=np.array([0.04761905, 0.04761905]),
        a=np.array([1.0, -0.9047619]),
        input_names=["input"],
        output_names=["output"],
    )


@pytest.fixture
def model_nk():
    return PolynomialModel(b=np.array([0.04761905, 0.04761905]), a=np.array([1.0, -0.9047619]), nk=2)


def test_init():
    poly = PolynomialModel()
    assert isinstance(poly, LTIModel)
    assert isinstance(poly, PolynomialModel)
    assert poly.Ts == 1.0


def test_simulate(model):
    u = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    y = model.simulate(u)
    assert y.shape == (20, 1)
    np.testing.assert_allclose(
        y.ravel(),
        [
            0.04761905,
            0.138322,
            0.22038657,
            0.29463547,
            0.36181304,
            0.42259275,
            0.47758392,
            0.52733783,
            0.57235327,
            0.61308153,
            0.64993091,
            0.68327083,
            0.71343551,
            0.74072736,
            0.76542,
            0.78776095,
            0.80797419,
            0.82626236,
            0.84280881,
            0.8577794,
        ],
    )


def test_simulate_nk(model_nk):
    u = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    y = model_nk.simulate(u)
    assert y.shape == (20, 1)
    np.testing.assert_allclose(
        y.ravel(),
        [
            0.0,
            0.0,
            0.04761905,
            0.138322,
            0.22038657,
            0.29463547,
            0.36181304,
            0.42259275,
            0.47758392,
            0.52733783,
            0.57235327,
            0.61308153,
            0.64993091,
            0.68327083,
            0.71343551,
            0.74072736,
            0.76542,
            0.78776095,
            0.80797419,
            0.82626236,
        ],
    )


def test_to_tf(model):
    tf = model.to_tf()
    print(tf)
    assert isinstance(tf, scipy.signal.TransferFunction)


def test_vectorize(model):
    theta = model.vectorize()
    np.testing.assert_allclose(theta, [0.04761905, 0.04761905, -0.9047619])


def test_reshape(model):
    model.reshape([0.05, 0.05, -0.9])
    np.testing.assert_allclose(model.b, [0.05, 0.05])
    np.testing.assert_allclose(model.a, [1.0, -0.9])


def test_repr_str(model):
    s = "PolynomialModel with Ts=1.0\ninput(s): ['input']\noutput(s): ['output']\nb:\n[0.04761905 0.04761905]\na:\n[ 1.        -0.9047619]\n"
    assert model.__repr__() == s
    assert model.__str__() == s


def test_impulse(model):
    ti, i = model.impulse_response()
    np.testing.assert_allclose(
        i.ravel()[:10],
        [
            4.76190500e-02,
            9.07029522e-02,
            8.20645753e-02,
            7.42489011e-02,
            6.71775768e-02,
            6.07797120e-02,
            5.49911678e-02,
            4.97539134e-02,
            4.50154452e-02,
            4.07282598e-02,
        ],
    )


def test_step(model):
    t_step, y_step = model.step_response()
    print(y_step)
    np.testing.assert_allclose(
        y_step.ravel()[:10],
        [
            0.04761905,
            0.138322,
            0.22038658,
            0.29463548,
            0.36181306,
            0.42259277,
            0.47758394,
            0.52733785,
            0.57235329,
            0.61308155,
        ],
    )


def test_frequency(model):
    omega, H = model.frequency_response()
    print(H)
    np.testing.assert_allclose(
        H.ravel()[:10],
        [
            9.99900010e-01 - 0.009999j,
            9.99840040e-01 - 0.01264653j,
            9.99744111e-01 - 0.0159945j,
            9.99590676e-01 - 0.02022762j,
            9.99345300e-01 - 0.02557873j,
            9.98952983e-01 - 0.03234071j,
            9.98325970e-01 - 0.04088066j,
            9.97324470e-01 - 0.05165628j,
            9.95726378e-01 - 0.06523311j,
            9.93180262e-01 - 0.08229963j,
        ],
    )
