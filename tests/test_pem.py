#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi.pem import OE, PEM


def test_pem_ss(data_siso_deterministic, ss_mod):
    identifyer = PEM(
        data_siso_deterministic,
        "y",
        "u",
        settings={"init": "po-moesp", "minimizer_kwargs": {"method":"BFGS"}},
    )
    mod = identifyer.ident(2)
    mod = mod.to_controllable_form()
    print(mod.A)
    np.testing.assert_allclose(mod.A, ss_mod.A, rtol=1e-3, atol=1e-3)


def test_oe(data_siso_deterministic_stochastic, poly_mod):
    identifyer = OE(
        data_siso_deterministic_stochastic,
        "y",
        "u",
        settings={"minimizer_kwargs": {"method":"BFGS"}},
    )
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, poly_mod.a, rtol=1e-1, atol=1e-1)
