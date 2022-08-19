#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

import numpy as np

from llsi.firor import FIROR


def test_firos(data_siso_deterministic):
    identifyer = FIROR(data_siso_deterministic, "y", "u")
    mod = identifyer.ident(2)
    mod = mod.to_controllable_form()
    print(mod.A)
    np.testing.assert_allclose(mod.A, [[1.6, -0.64], [1.0, 0]], rtol=1e-3, atol=1e-3)
