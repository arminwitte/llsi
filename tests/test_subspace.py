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
    identifyer = N4SID(data_mimo_deterministic, ["y0", "y1"], ["u0", "u1"])
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
    t, i = mod.impulse()
    assert i.shape == (100,2)
    np.testing.assert_allclose(i[:10,:],[[0],[0]])
    
