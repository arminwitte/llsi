#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi.subspace import N4SID, PO_MOESP


def test_n4sid(data_siso_deterministic, ss_mod):
    identifyer = N4SID(data_siso_deterministic, "y", "u")
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [1.44074715e03, 7.11128485e02, 2.03261693e-01, 3.69458451e-09, 3.36570175e-09],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mod.to_controllable_form().A, ss_mod.A, rtol=1e-1, atol=1e-1
    )


def test_po_moesp(data_siso_deterministic, ss_mod):
    identifyer = PO_MOESP(data_siso_deterministic, "y", "u")
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [
            1.43920126e03,
            7.10890502e02,
            2.03261254e-01,
            2.97663946e-09,
            2.06566781e-09,
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mod.to_controllable_form().A, ss_mod.A, rtol=1e-3, atol=1e-3
    )


def test_n4sid_mimo(data_mimo_deterministic):
    # data_mimo_deterministic.equidistant()
    identifyer = N4SID(data_mimo_deterministic, ["y0", "y1"], ["u0", "u1"])
    sys0 = N4SID(data_mimo_deterministic, "y0", "u0").ident(1)
    sys1 = N4SID(data_mimo_deterministic, "y1", "u1").ident(1)
    mod = identifyer.ident(1)
    print(mod.info["Hankel singular values"])
    # np.testing.assert_allclose(
    #     mod.info["Hankel singular values"],
    #     [
    #         6.40829670e00,
    #         3.94039974e-04,
    #         2.47748028e-08,
    #         1.01130860e-08,
    #         9.63664129e-09,
    #         7.82223423e-09,
    #         7.34439653e-09,
    #         7.30785370e-09,
    #         6.41191499e-09,
    #         5.96869347e-09,
    #         5.92269632e-09,
    #         5.38759994e-09,
    #         5.29767778e-09,
    #         5.02836434e-09,
    #         4.37323782e-09,
    #         4.03590185e-09,
    #         3.79029097e-09,
    #         3.32988710e-09,
    #         2.74121403e-09,
    #         2.40889178e-09,
    #     ],
    #     rtol=1e-6,
    #     atol=1e-6,
    # )

    # assert mod.A.shape == (2, 2)
    # assert mod.B.shape == (2, 2)
    # assert mod.C.shape == (2, 2)
    # assert mod.D.shape == (2, 2)

    print(mod)

    # np.testing.assert_allclose(
    #    mod.to_controllable_form().A, ss_mod.A, rtol=1e-1, atol=1e-1
    # )

    print("step response sys0:\n", sys0.step_response(10)[1])
    print("step response sys1:\n", sys1.step_response(10)[1])
    t, i = mod.step_response(10)
    # assert i.shape == (10, 2)
    print("step response mod :\n", i)
    np.testing.assert_allclose(i, [[0], [0]])

    # d = data_mimo_deterministic
    # # y = mod.simulate(np.array([d["u0"], d["u1"]]).T)
    # y = mod.simulate([d["u0"], d["u1"]])
    # y_val = np.array([d["y0"], d["y1"]]).T
    # np.testing.assert_allclose(y[1000:1010, :], y_val[1000:1010, :])


def test_po_moesp_mimo(data_mimo_deterministic):
    # data_mimo_deterministic.equidistant()
    identifyer = PO_MOESP(data_mimo_deterministic, ["y0", "y1"], ["u0", "u1"])
    # sys0 = PO_MOESP(data_mimo_deterministic, "y0", "u0").ident(1)
    # sys1 = PO_MOESP(data_mimo_deterministic, "y1", "u1").ident(1)
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    # np.testing.assert_allclose(
    #     mod.info["Hankel singular values"],
    #     [
    #         6.40829670e00,
    #         3.94039974e-04,
    #         2.47748028e-08,
    #         1.01130860e-08,
    #         9.63664129e-09,
    #         7.82223423e-09,
    #         7.34439653e-09,
    #         7.30785370e-09,
    #         6.41191499e-09,
    #         5.96869347e-09,
    #         5.92269632e-09,
    #         5.38759994e-09,
    #         5.29767778e-09,
    #         5.02836434e-09,
    #         4.37323782e-09,
    #         4.03590185e-09,
    #         3.79029097e-09,
    #         3.32988710e-09,
    #         2.74121403e-09,
    #         2.40889178e-09,
    #     ],
    #     rtol=1e-6,
    #     atol=1e-6,
    # )

    # assert mod.A.shape == (2, 2)
    # assert mod.B.shape == (2, 2)
    # assert mod.C.shape == (2, 2)
    # assert mod.D.shape == (2, 2)

    print(mod)

    # np.testing.assert_allclose(
    #    mod.to_controllable_form().A, ss_mod.A, rtol=1e-1, atol=1e-1
    # )

    print("step response sys0:\n", sys0.step_response(10)[1])
    print("step response sys1:\n", sys1.step_response(10)[1])
    t, i = mod.step_response(10)
    # assert i.shape == (10, 2)
    print("step response mod :\n", i)
    np.testing.assert_allclose(i, [[0], [0]])

    # d = data_mimo_deterministic
    # # y = mod.simulate(np.array([d["u0"], d["u1"]]).T)
    # y = mod.simulate([d["u0"], d["u1"]])
    # y_val = np.array([d["y0"], d["y1"]]).T
    # np.testing.assert_allclose(y[1000:1010, :], y_val[1000:1010, :])


