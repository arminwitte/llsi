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


def test_n4sid_mimo(data_mimo_deterministic, ss_mod):
    data_mimo_deterministic.equidistant()
    identifyer = N4SID(data_mimo_deterministic, ["y0", "y1"], ["u0", "u1"])
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [
            6.40829670e00,
            3.94039974e-04,
            2.47748028e-08,
            1.01130860e-08,
            9.63664129e-09,
            7.82223423e-09,
            7.34439653e-09,
            7.30785370e-09,
            6.41191499e-09,
            5.96869347e-09,
            5.92269632e-09,
            5.38759994e-09,
            5.29767778e-09,
            5.02836434e-09,
            4.37323782e-09,
            4.03590185e-09,
            3.79029097e-09,
            3.32988710e-09,
            2.74121403e-09,
            2.40889178e-09,
        ],
        rtol=1e-6,
        atol=1e-6,
    )


    assert mod.A.shape == (2,2)
    assert mod.B.shape == (2,2)
    assert mod.C.shape == (2,2)
    assert mod.D.shape == (2,2)
    
    # np.testing.assert_allclose(
    #    mod.to_controllable_form().A, ss_mod.A, rtol=1e-1, atol=1e-1
    # )
    
    t, i = mod.impulse_response()
    assert i.shape == (100, 2)
    np.testing.assert_allclose(i[:10, :], [[0], [0]])

    d = data_mimo_deterministic
    y = mod.simulate([d["u0"], d["u1"]])
    np.testing.assert_allclose(y[:10, :], [[0], [0]])

