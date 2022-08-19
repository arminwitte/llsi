#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi.pem import OE, PEM


def test_pem_ss(data_siso_deterministic):
    identifyer = PEM(data_siso_deterministic, "y", "u", settings={"init": "po-moesp"})
    mod = identifyer.ident(2)
    mod = mod.to_controllable_form()
    print(mod.A)
    np.testing.assert_allclose(mod.A, [[1.6, -0.64], [1.0, 0]], rtol=1e-3, atol=1e-3)


def test_oe(data_siso_deterministic_stochastic):
    identifyer = OE(data_siso_deterministic_stochastic, "y", "u")
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, [1.0, -1.6, 0.64], rtol=1e-1, atol=1e-1)
