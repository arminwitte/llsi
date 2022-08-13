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
from llsi.polynomialmodel import PolynomialModel


@pytest.fixture
def model():
    return PolynomialModel(
        b=np.array([0.04761905, 0.04761905]), a=np.array([1.0, -0.9047619])
    )


@pytest.fixture
def model_nk():
    return PolynomialModel(
        b=np.array([0.04761905, 0.04761905]), a=np.array([1.0, -0.9047619]), nk=2
    )


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
    np.testing.assert_allclose(
        y,
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
    np.testing.assert_allclose(
        y,
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