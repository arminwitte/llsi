#!/usr/bin/env python3
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np
import pytest

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
    np.testing.assert_allclose(mod.to_controllable_form().A, ss_mod.A, rtol=1e-1, atol=1e-1)


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
    np.testing.assert_allclose(mod.to_controllable_form().A, ss_mod.A, rtol=1e-3, atol=1e-3)


def test_n4sid_mimo(data_mimo_deterministic):
    with pytest.raises(NotImplementedError):
        identifyer = N4SID(data_mimo_deterministic, ["y0", "y1"], ["u0", "u1"])

    return
    sys0 = N4SID(data_mimo_deterministic, "y0", "u0").ident(1)
    sys1 = N4SID(data_mimo_deterministic, "y1", "u1").ident(1)
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [
            5.23238711e01,
            2.46007773e01,
            9.15587701e-02,
            9.90325106e-09,
            9.65156298e-09,
            8.85547497e-09,
            8.51780379e-09,
            8.32136576e-09,
            7.44491698e-09,
            7.03837731e-09,
            5.99893485e-09,
            5.85910395e-09,
            5.52005230e-09,
            5.26083503e-09,
            5.10508405e-09,
            4.67083640e-09,
            3.59163096e-09,
            3.53484111e-09,
            3.01876150e-09,
            2.64576961e-09,
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    assert mod.A.shape == (2, 2)
    assert mod.B.shape == (2, 2)
    assert mod.C.shape == (2, 2)
    assert mod.D.shape == (2, 2)

    print(mod)

    t, i = mod.step_response(10)
    assert i.shape == (10, 2)
    np.testing.assert_allclose(
        i,
        np.array([sys0.step_response(10)[1].ravel(), sys1.step_response(10)[1].ravel()]).T,
        rtol=1e-5,
        atol=1e-5,
    )


def test_po_moesp_mimo(data_mimo_deterministic):
    identifyer = PO_MOESP(data_mimo_deterministic, ["y0", "y1"], ["u0", "u1"])
    sys0 = PO_MOESP(data_mimo_deterministic, "y0", "u0").ident(1)
    sys1 = PO_MOESP(data_mimo_deterministic, "y1", "u1").ident(1)
    mod = identifyer.ident(2)
    print(mod.info["Hankel singular values"])
    np.testing.assert_allclose(
        mod.info["Hankel singular values"],
        [
            5.13375445e01,
            2.45203750e01,
            9.15569690e-02,
            9.88027549e-09,
            9.63367326e-09,
            8.84678201e-09,
            8.50474100e-09,
            8.31302996e-09,
            7.42880630e-09,
            7.02472812e-09,
            5.99054895e-09,
            5.85392065e-09,
            5.51099680e-09,
            5.25372488e-09,
            5.10043257e-09,
            4.66624062e-09,
            3.58670073e-09,
            3.52980561e-09,
            3.01474135e-09,
            2.64218261e-09,
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    assert mod.A.shape == (2, 2)
    assert mod.B.shape == (2, 2)
    assert mod.C.shape == (2, 2)
    assert mod.D.shape == (2, 2)

    print(mod)

    t, i = mod.step_response(10)
    assert i.shape == (10, 2)
    np.testing.assert_allclose(
        i,
        np.array([sys0.step_response(10)[1].ravel(), sys1.step_response(10)[1].ravel()]).T,
        rtol=1e-5,
        atol=1e-5,
    )
