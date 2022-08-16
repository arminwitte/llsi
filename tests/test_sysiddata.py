#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:25:46 2021

@author: armin
"""

import numpy as np
import pytest

from llsi.sysiddata import SysIdData


@pytest.fixture
def data():
    data = SysIdData(
        y=[1, 2, 3, 4, 5, 6], u=[1, 4, 9, 16, 25, 36], t=[1, 1.5, 2.5, 3.0, 3.3, 4.1]
    )
    return data


def test_init():
    data = SysIdData(
        y=[1, 2, 3, 4, 5, 6], u=[1, 4, 9, 16, 25, 36], t=[1, 1.5, 2.5, 3.0, 3.3, 4.1]
    )
    np.testing.assert_equal(data["y"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_equal(data["u"], [1, 4, 9, 16, 25, 36])
    np.testing.assert_equal(data.time(), [1, 1.5, 2.5, 3.0, 3.3, 4.1])


def test_equidistant(data):
    data.equidistant()
    np.testing.assert_allclose(data["y"], [1.0, 2.12, 2.74, 3.72, 5.225, 6.0])
    np.testing.assert_allclose(data["u"], [1.0, 4.6, 7.7, 14.04, 27.475, 36.0])
    np.testing.assert_allclose(data.time(), [1.0, 1.62, 2.24, 2.86, 3.48, 4.1])


def test_center(data):
    data.equidistant()
    data.center()
    np.testing.assert_allclose(
        data["y"], [-2.4675, -1.3475, -0.7275, 0.2525, 1.7575, 2.5325]
    )
    np.testing.assert_allclose(
        data["u"],
        [-14.135833, -10.535833, -7.435833, -1.095833, 12.339167, 20.864167],
        rtol=1e-6,
    )


def test_crop(data):
    data.crop(1, -1)
    np.testing.assert_equal(data["y"], [2, 3, 4, 5])
    np.testing.assert_equal(data["u"], [4, 9, 16, 25])
    np.testing.assert_equal(data.time(), [1.5, 2.5, 3.0, 3.3])


def test_generate_prbs():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    np.testing.assert_allclose(t[29:32], [29.0, 30.0, 31.0])
    np.testing.assert_allclose(u[29:32], [1.0, 0.0, 1.0])


def test_downsample():
    t, u = SysIdData.generate_prbs(1000, 1.0)
    data = SysIdData(t=t, u=u)
    data.equidistant()
    data.downsample(2)
    assert data.Ts == 2.0
    np.testing.assert_allclose(data.time()[14:17], [28.0, 30.0, 32.0])
    np.testing.assert_allclose(
        data["u"][14:17], [0.206036, 0.59716, 0.923016], rtol=1e-6
    )


# add_series
# remove
# plot
# show
