#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi.arx import ARX


def test_arx(data_siso_deterministic):
    identifyer = ARX(data_siso_deterministic, "y", "u")
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, [1.0, -1.6, 0.64], rtol=1e-5, atol=1e-5)


def test_arx_regul(data_siso_deterministic):
    identifyer = ARX(data_siso_deterministic, "y", "u", settings={"lambda": 1e-3})
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, [1.0, -1.6, 0.64], rtol=1e-5, atol=1e-5)


def test_arx_cov(data_siso_deterministic_stochastic):
    identifyer = ARX(
        data_siso_deterministic_stochastic, "y", "u", settings={"lambda": 1e-3}
    )
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, [1.0, -0.76, -0.16], rtol=1e-1, atol=1e-1)
    print(mod.cov)
    np.testing.assert_allclose(
        mod.cov,
        [
            [
                2.79082789e-04,
                -6.07163602e-05,
                -1.87550336e-06,
                -1.54457406e-06,
                3.12059702e-05,
            ],
            [
                -6.07163602e-05,
                2.59428258e-04,
                -1.44226080e-05,
                9.22973109e-06,
                -3.49372240e-05,
            ],
            [
                -1.87550336e-06,
                -1.44226080e-05,
                4.77281086e-05,
                4.37283115e-05,
                2.34874306e-05,
            ],
            [
                -1.54457406e-06,
                9.22973109e-06,
                4.37283115e-05,
                4.89017693e-05,
                -1.52653273e-05,
            ],
            [
                3.12059702e-05,
                -3.49372240e-05,
                2.34874306e-05,
                -1.52653273e-05,
                2.05893962e-04,
            ],
        ],
        rtol=1e-1,
        atol=1e-1,
    )


def test_arx_pinv(data_siso_deterministic_stochastic):
    identifyer = ARX(
        data_siso_deterministic_stochastic,
        "y",
        "u",
        settings={"lstsq_method": "pinv"},
    )
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, [1.0, -0.76, -0.16], rtol=1e-1, atol=1e-1)
    print(mod.cov)
    np.testing.assert_allclose(
        mod.cov,
        [
            [
                2.79082789e-04,
                -6.07163602e-05,
                -1.87550336e-06,
                -1.54457406e-06,
                3.12059702e-05,
            ],
            [
                -6.07163602e-05,
                2.59428258e-04,
                -1.44226080e-05,
                9.22973109e-06,
                -3.49372240e-05,
            ],
            [
                -1.87550336e-06,
                -1.44226080e-05,
                4.77281086e-05,
                4.37283115e-05,
                2.34874306e-05,
            ],
            [
                -1.54457406e-06,
                9.22973109e-06,
                4.37283115e-05,
                4.89017693e-05,
                -1.52653273e-05,
            ],
            [
                3.12059702e-05,
                -3.49372240e-05,
                2.34874306e-05,
                -1.52653273e-05,
                2.05893962e-04,
            ],
        ],
        rtol=1e-1,
        atol=1e-1,
    )


def test_arx_lstsq(data_siso_deterministic_stochastic):
    identifyer = ARX(
        data_siso_deterministic_stochastic,
        "y",
        "u",
        settings={"lstsq_method": "lstsq"},
    )
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, [1.0, -0.76, -0.16], rtol=1e-1, atol=1e-1)
    print(mod.cov)
    np.testing.assert_allclose(
        mod.cov,
        [
            [
                2.79082789e-04,
                -6.07163602e-05,
                -1.87550336e-06,
                -1.54457406e-06,
                3.12059702e-05,
            ],
            [
                -6.07163602e-05,
                2.59428258e-04,
                -1.44226080e-05,
                9.22973109e-06,
                -3.49372240e-05,
            ],
            [
                -1.87550336e-06,
                -1.44226080e-05,
                4.77281086e-05,
                4.37283115e-05,
                2.34874306e-05,
            ],
            [
                -1.54457406e-06,
                9.22973109e-06,
                4.37283115e-05,
                4.89017693e-05,
                -1.52653273e-05,
            ],
            [
                3.12059702e-05,
                -3.49372240e-05,
                2.34874306e-05,
                -1.52653273e-05,
                2.05893962e-04,
            ],
        ],
        rtol=1e-1,
        atol=1e-1,
    )


def test_arx_qr(data_siso_deterministic_stochastic):
    identifyer = ARX(
        data_siso_deterministic_stochastic,
        "y",
        "u",
        settings={"lstsq_method": "qr"},
    )
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, [1.0, -0.76, -0.16], rtol=1e-1, atol=1e-1)
    print(mod.cov)
    np.testing.assert_allclose(
        mod.cov,
        [
            [
                2.79082789e-04,
                -6.07163602e-05,
                -1.87550336e-06,
                -1.54457406e-06,
                3.12059702e-05,
            ],
            [
                -6.07163602e-05,
                2.59428258e-04,
                -1.44226080e-05,
                9.22973109e-06,
                -3.49372240e-05,
            ],
            [
                -1.87550336e-06,
                -1.44226080e-05,
                4.77281086e-05,
                4.37283115e-05,
                2.34874306e-05,
            ],
            [
                -1.54457406e-06,
                9.22973109e-06,
                4.37283115e-05,
                4.89017693e-05,
                -1.52653273e-05,
            ],
            [
                3.12059702e-05,
                -3.49372240e-05,
                2.34874306e-05,
                -1.52653273e-05,
                2.05893962e-04,
            ],
        ],
        rtol=1e-1,
        atol=1e-1,
    )
