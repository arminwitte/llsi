#! /usr/bin/env python3
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi import sysid

# Deterministic
# =============

# State Space
# -----------


def test_n4sid_deterministic(data_siso_deterministic, ss_mod):
    mod = sysid(data_siso_deterministic, "y", "u", (2), method="n4sid")

    # parameters
    mod_ = mod.to_controllable_form()
    np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-2, atol=1e-2)

    # HSV
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [1.44074715e03, 7.11128485e02, 2.03261693e-01, 3.81475198e-09, 9.21749364e-10],
        rtol=1e-6,
        atol=1e-6,
    )

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = ss_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e0, atol=1e0)


def test_po_moesp_deterministic(data_siso_deterministic, ss_mod):
    mod = sysid(data_siso_deterministic, "y", "u", (2), method="po-moesp")

    # parameters
    mod_ = mod.to_controllable_form()
    np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-5, atol=1e-5)

    # HSV
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [
            1.43920126e03,
            7.10890502e02,
            2.03261254e-01,
            3.91510332e-09,
            2.24387529e-09,
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = ss_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e-5, atol=1e-5)


# def test_pem_ss_deterministic(data_siso_deterministic, ss_mod):
#     mod = sysid(
#         data_siso_deterministic,
#         "y",
#         "u",
#         (2),
#         method="pem",
#         settings={"init": "po-moesp"},
#     )

#     # parameters
#     mod_ = mod.to_controllable_form()
#     # np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-5, atol=1e-5)

#     # impulse response
#     ti, i = mod.impulse_response()
#     ti_, i_ = ss_mod.impulse_response()
#     np.testing.assert_allclose(i, i_, rtol=1e-5, atol=1e-5)


def test_firor_deterministic(data_siso_deterministic, ss_mod):
    mod = sysid(
        data_siso_deterministic,
        "y",
        "u",
        (2),
        method="firor",
        settings={"lambda": 1e0, "fir_order": 50},
    )

    # parameters
    mod_ = mod.to_controllable_form()
    np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-3, atol=1e-3)

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = ss_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e-2, atol=1e-2)


# Polynomial
# ----------


def test_arx_deterministic(data_siso_deterministic, poly_mod):
    mod = sysid(data_siso_deterministic, "y", "u", (2, 3, 0), method="arx")

    # parameters
    np.testing.assert_allclose(mod.a, poly_mod.a, rtol=1e-5, atol=1e-5)

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = poly_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e-5, atol=1e-5)


def test_fir_deterministic(data_siso_deterministic, poly_mod):
    mod = sysid(data_siso_deterministic, "y", "u", (0, 100, 0), method="fir")

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = poly_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e-4, atol=1e-4)


def test_oe_deterministic(data_siso_deterministic, poly_mod):
    mod = sysid(
        data_siso_deterministic,
        "y",
        "u",
        (2, 3, 0),
        method="oe",
        settings={"minimizer_kwargs": {"method": "powell"}},
    )

    # parameters
    np.testing.assert_allclose(mod.a, poly_mod.a, rtol=1e-5, atol=1e-5)

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = poly_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e-3, atol=1e-3)


# Deterministic plus Stochastic
# =============

# State Space
# -----------


def test_n4sid_deterministic_stochastic(data_siso_deterministic_stochastic, ss_mod):
    mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2), method="n4sid")

    # parameters
    mod_ = mod.to_controllable_form()
    np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-2, atol=1e-2)

    # HSV
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [1.44074715e03, 7.11128485e02, 2.03261693e-01, 3.81475198e-09, 9.21749364e-10],
        rtol=2e-0,
        atol=2e-0,
    )

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = ss_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e1, atol=1e1)


def test_po_moesp_deterministic_stochastic(data_siso_deterministic_stochastic, ss_mod):
    mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2), method="po-moesp")

    # parameters
    mod_ = mod.to_controllable_form()
    np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-3, atol=1e-3)

    # HSV
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [1439.52436941, 710.90138305, 2.38824865, 1.93589346, 1.51688614],
        rtol=1e-3,
        atol=1e-3,
    )

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = ss_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=5e-2, atol=5e-2)


# def test_pem_ss_deterministic_stochastic(data_siso_deterministic_stochastic, ss_mod):
#     mod = sysid(
#         data_siso_deterministic_stochastic,
#         "y",
#         "u",
#         (2),
#         method="pem",
#         settings={"init": "po-moesp"},
#     )

#     # parameters
#     mod_ = mod.to_controllable_form()
#     # np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-5, atol=1e-5)

#     # impulse response
#     ti, i = mod.impulse_response()
#     ti_, i_ = ss_mod.impulse_response()
#     np.testing.assert_allclose(i, i_, rtol=1e-5, atol=1e-5)


def test_firor_deterministic_stochastic(data_siso_deterministic_stochastic, ss_mod):
    mod = sysid(
        data_siso_deterministic_stochastic,
        "y",
        "u",
        (2),
        method="firor",
        settings={"lambda": 1e0, "fir_order": 50},
    )

    # parameters
    mod_ = mod.to_controllable_form()
    np.testing.assert_allclose(mod_.A, ss_mod.A, rtol=1e-3, atol=1e-3)

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = ss_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e-2, atol=1e-2)


# Polynomial
# ----------


def test_arx_deterministic_stochastic(data_siso_deterministic_stochastic, poly_mod):
    mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2, 3, 0), method="arx")

    # parameters
    np.testing.assert_allclose(mod.a, poly_mod.a, rtol=5e-2, atol=5e-2)

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = poly_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=1e-0, atol=1e-0)


def test_fir_deterministic_stochastic(data_siso_deterministic_stochastic, poly_mod):
    mod = sysid(data_siso_deterministic_stochastic, "y", "u", (0, 100, 0), method="fir")

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = poly_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=5e-2, atol=5e-2)


def test_oe_deterministic_stochastic(data_siso_deterministic_stochastic, poly_mod):
    mod = sysid(
        data_siso_deterministic_stochastic,
        "y",
        "u",
        (2, 3, 0),
        method="oe",
        settings={"minimizer_kwargs": {"method": "nelder-mead"}},
    )

    # parameters
    np.testing.assert_allclose(mod.a, poly_mod.a, rtol=1e-2, atol=1e-2)

    # impulse response
    ti, i = mod.impulse_response()
    ti_, i_ = poly_mod.impulse_response()
    np.testing.assert_allclose(i, i_, rtol=5e-2, atol=5e-2)


def test_poly_input_output_name(data_siso_deterministic, poly_mod):
    mod = sysid(data_siso_deterministic, "y", "u", (0, 100, 0), method="fir")
    assert mod.input_names == "u"
    assert mod.output_names == "y"


def test_ss_input_output_name(data_mimo_deterministic, poly_mod):
    mod = sysid(data_mimo_deterministic, ["y0", "y1"], ["u0", "u1"], 2, method="po-moesp")
    assert mod.input_names == ["u0", "u1"]
    assert mod.output_names == ["y0", "y1"]


#######################################################################################


# def test_po_moesp_deterministic(data_siso_deterministic):
#     mod = sysid(data_siso_deterministic, "y", "u", (2), method="po-moesp")
#     # np.testing.assert_allclose(mod.info["Hankel singular values"], [0.12, 0.016, 0., 0., 0.], rtol=1e-2, atol=1e-2)
#     ti, i = mod.impulse_response()
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)
#     np.testing.assert_allclose(
#         mod.to_controllable_form().A, [[1.6, -0.64], [1.0, 0.0]], rtol=1e-3, atol=1e-3
#     )


# def test_pem_ss_deterministic(data_siso_deterministic):
#     mod = sysid(
#         data_siso_deterministic,
#         "y",
#         "u",
#         (2),
#         method="pem",
#         settings={"init": "po-moesp"},
#     )
#     ti, i = mod.impulse_response()
#     print(mod.cov)
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)


# def test_pem_poly_deterministic(data_siso_deterministic):
#     mod = sysid(
#         data_siso_deterministic,
#         "y",
#         "u",
#         (1, 2, 0),
#         method="pem",
#         settings={"init": "arx"},
#     )
#     ti, i = mod.impulse_response()
#     print(mod.cov)
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)


# def test_oe_deterministic(data_siso_deterministic):
#     mod = sysid(data_siso_deterministic, "y", "u", (1, 2, 0), method="oe")
#     ti, i = mod.impulse_response()
#     print(mod.cov)
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)


# def test_arx_deterministic(data_siso_deterministic):
#     mod = sysid(
#         data_siso_deterministic,
#         "y",
#         "u",
#         (2, 3, 0),
#         method="arx",
#         settings={"lstsq_method": "svd", "lambda": 1e-9},
#     )
#     ti, i = mod.impulse_response()
#     print(mod.cov)
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)


# def test_fir_deterministic(data_siso_deterministic):
#     mod = sysid(
#         data_siso_deterministic,
#         "y",
#         "u",
#         (0, 50, 0),
#         method="fir",
#         settings={"lstsq_method": "svd", "lambda": 1e0},
#     )
#     ti, i = mod.impulse_response()
#     # import matplotlib.pyplot as plt
#     # plt.show()
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=1e-3, atol=1e-3)


# def test_n4sid_deterministic_stochastic(data_siso_deterministic_stochastic):
#     mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2), method="n4sid")
#     ti, i = mod.impulse_response()
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)


# def test_po_moesp_deterministic_stochastic(data_siso_deterministic_stochastic):
#     mod = sysid(data_siso_deterministic_stochastic, "y", "u", (2), method="po-moesp")
#     ti, i = mod.impulse_response()
#     # import matplotlib.pyplot as plt
#     # plt.show()
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)


# def test_pem_ss_deterministic_stochastic(data_siso_deterministic_stochastic):
#     mod = sysid(
#         data_siso_deterministic_stochastic,
#         "y",
#         "u",
#         (2),
#         method="pem",
#         settings={"init": "po-moesp"},
#     )
#     ti, i = mod.impulse_response()
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)


# def test_oe_deterministic_stochastic(data_siso_deterministic_stochastic):
#     mod = sysid(data_siso_deterministic_stochastic, "y", "u", (1, 2, 0), method="oe")
#     ti, i = mod.impulse_response()
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)


# def test_arx_deterministic_stochastik(data_siso_deterministic_stochastic):
#     mod = sysid(
#         data_siso_deterministic_stochastic,
#         "y",
#         "u",
#         (2, 3, 0),
#         method="arx",
#         settings={"lstsq_method": "svd", "lambda": 1e-3},
#     )
#     ti, i = mod.impulse_response()
#     # print(mod.cov)
#     np.testing.assert_allclose(i[:3], [0.0, 1.0, 1.6], rtol=0.1, atol=0.1)
