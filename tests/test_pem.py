#!/usr/bin/env python3
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi.pem import ADAM

# def test_pem_ss(data_siso_deterministic, ss_mod):
#     identifyer = PEM(
#         data_siso_deterministic,
#         "y",
#         "u",
#         settings={"init": "po-moesp", "minimizer_kwargs": {"method": "BFGS"}},
#     )
#     mod = identifyer.ident(2)
#     mod = mod.to_controllable_form()
#     print(mod.A)
#     np.testing.assert_allclose(mod.A, ss_mod.A, rtol=1e-3, atol=1e-3)


def test_adam_ss(data_siso_deterministic, ss_mod):
    identifyer = ADAM(
        data_siso_deterministic,
        "y",
        "u",
        settings={"init": "po-moesp", "learning_rate": 1e-5},
    )
    mod = identifyer.ident(2)
    mod = mod.to_controllable_form()
    print(mod.A)
    np.testing.assert_allclose(mod.A, ss_mod.A, rtol=5e-2, atol=5e-2)


# def test_oe(data_siso_deterministic_stochastic, poly_mod):
#     identifyer = OE(
#         data_siso_deterministic_stochastic,
#         "y",
#         "u",
#         settings={"minimizer_kwargs": {"method": "BFGS"}},
#     )
#     mod = identifyer.ident((2, 3, 0))
#     print(mod.a)
#     np.testing.assert_allclose(mod.a, poly_mod.a, rtol=1e-1, atol=1e-1)


def test_adam_oe_noise(data_siso_deterministic_stochastic, poly_mod):
    identifyer = ADAM(
        data_siso_deterministic_stochastic,
        "y",
        "u",
        settings={"minimizer_kwargs": {"method": "BFGS"}},
    )
    mod = identifyer.ident((2, 3, 0))
    print(mod.a)
    np.testing.assert_allclose(mod.a, poly_mod.a, rtol=0.5, atol=0.5)
