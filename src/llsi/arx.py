#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 01:34:47 2022

@author: armin
"""

import numpy as np
import scipy.linalg

from .polynomialmodel import PolynomialModel
from .sysidalgbase import SysIdAlgBase


class ARX(SysIdAlgBase):
    def __init__(self, data, y_name, u_name):
        super().__init__(data, y_name, u_name)

    def ident(self, order):
        na, nb, nk = order
        Phi, y = self._observations(na, nb, nk)

        theta = scipy.linalg.pinv(Phi) @ y
        # theta, res, rank, s = scipy.linalg.lstsq(Phi, y)

        # print(theta)

        b = theta[:nb]
        a = np.hstack(([1.0], theta[nb:]))

        mod = PolynomialModel(b=b, a=a, nk=nk, Ts=self.Ts)

        return mod

    def _observations(self, na, nb, nk):
        nn = max(nb + nk, na)
        N = self.u.ravel().shape[0]
        Phi = np.empty((N - nn, nb + na))
        for i in range(nb):
            Phi[:, i] = self.u[nn - i - nk : N - i - nk]
        for i in range(na):
            Phi[:, nb + i] = -self.y[nn - i - 1 : N - i - 1]

        y = self.y[nn:N]

        return Phi, y

    @staticmethod
    def name():
        return "arx"
