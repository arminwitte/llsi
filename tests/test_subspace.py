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
    # First two singular values should be large, rest should be small
    hsv = mod.info["Hankel singular values"]
    assert hsv[0] > 1000
    assert hsv[1] > 500
    assert hsv[2] < 1
    assert np.all(hsv[3:] < 1e-8)

    # Additional validation: HSV should be monotonically decreasing
    assert np.all(hsv[:-1] >= hsv[1:]), "Hankel singular values should be monotonically decreasing"

    # For a 2nd order system, the decay should be significant
    decay_ratio = hsv[2] / hsv[0]
    assert decay_ratio < 0.001, f"Decay ratio {decay_ratio} too high for 2nd order system"

    np.testing.assert_allclose(mod.to_controllable_form().A, ss_mod.A, rtol=1e-1, atol=1e-1)


def test_po_moesp(data_siso_deterministic, ss_mod):
    identifyer = PO_MOESP(data_siso_deterministic, "y", "u")
    mod = identifyer.ident(2)
    # First two singular values should be large, rest should be small
    hsv = mod.info["Hankel singular values"]
    assert hsv[0] > 1000
    assert hsv[1] > 500
    assert hsv[2] < 1
    assert np.all(hsv[3:] < 1e-8)

    # Additional validation: HSV should be monotonically decreasing
    assert np.all(hsv[:-1] >= hsv[1:]), "Hankel singular values should be monotonically decreasing"

    # For a 2nd order system, the decay should be significant
    decay_ratio = hsv[2] / hsv[0]
    assert decay_ratio < 0.001, f"Decay ratio {decay_ratio} too high for 2nd order system"

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
    # First two singular values should be large, rest should be small
    hsv = mod.info["Hankel singular values"]
    assert hsv[0] > 40
    assert hsv[1] > 20
    assert hsv[2] < 1
    assert np.all(hsv[3:] < 1e-8)

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


def test_enforce_stability():
    # Create an unstable matrix
    # Eigenvalues: 2, 0.5
    # A = V D V^-1
    D = np.diag([2.0, 0.5])
    V = np.array([[1, 1], [0, 1]])
    A = V @ D @ np.linalg.inv(V)

    # Stabilize
    from llsi.subspace import SubspaceIdent

    A_stable = SubspaceIdent.enforce_stability(A)

    vals = np.linalg.eigvals(A_stable)
    assert np.all(np.abs(vals) <= 1.0 + 1e-9)
    # The unstable eigenvalue 2 should become 0.99 (default radius).
    # The stable eigenvalue 0.5 should remain 0.5.

    vals_sorted = np.sort(np.abs(vals))
    np.testing.assert_allclose(vals_sorted, [0.5, 0.99])


def test_n4sid_enforce_stability(data_siso_deterministic):
    # Just check if it runs without error and returns a stable model
    identifyer = N4SID(data_siso_deterministic, "y", "u", settings={"enforce_stability": True})
    mod = identifyer.ident(2)
    vals = np.linalg.eigvals(mod.A)
    assert np.all(np.abs(vals) <= 1.0 + 1e-9)


def test_po_moesp_enforce_stability(data_siso_deterministic):
    # Just check if it runs without error and returns a stable model
    identifyer = PO_MOESP(data_siso_deterministic, "y", "u", settings={"enforce_stability": True})
    mod = identifyer.ident(2)
    vals = np.linalg.eigvals(mod.A)
    assert np.all(np.abs(vals) <= 1.0 + 1e-9)
