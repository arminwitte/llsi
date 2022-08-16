#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi.subspace import N4SID, PO_MOESP


def test_n4sid(data_siso_deterministic):
    identifyer = N4SID(data_siso_deterministic, "y", "u")
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [6.34829911e02, 7.96860814e01, 1.47245544e-01, 5.67247918e-09, 3.30006332e-09],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mod.to_controllable_form().A, [[1.6, -0.64], [1.0, 0.0]], rtol=1e-1, atol=1e-1
    )


def test_po_moesp(data_siso_deterministic):
    identifyer = PO_MOESP(data_siso_deterministic, "y", "u")
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [
            6.34705690e-02,
            7.96133156e-03,
            1.47245160e-05,
            4.94276398e-13,
            3.86270211e-13,
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mod.to_controllable_form().A, [[1.6, -0.64], [1.0, 0.0]], rtol=1e-3, atol=1e-3
    )
