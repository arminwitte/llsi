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
    def __init__(self, data, y_name, u_name, settings={}):
        super().__init__(data, y_name, u_name, settings=settings)

    def ident(self, order):
        na, nb, nk = order
        Phi, y = self._observations(na, nb, nk)

        # theta = scipy.linalg.pinv(Phi) @ y
        # theta, res, rank, s = scipy.linalg.lstsq(Phi, y)

        lstsq_method = self.settings.get("lstsq_method", "svd")
        l = self.settings.get("lambda", 0.0)
        if lstsq_method in "pinv":
            theta, cov = self._lstsq_pinv(Phi, y)
        elif lstsq_method in "lstsq":
            theta, cov = self._lstsq_lstsq(Phi, y)
        elif lstsq_method in "qr":
            theta, cov = self._lstsq_qr(Phi, y)
        elif lstsq_method in "svd":
            theta, cov = self._lstsq_svd(Phi, y, l)
        # print(theta)

        b = theta[:nb]
        a = np.hstack(([1.0], theta[nb:]))

        mod = PolynomialModel(b=b, a=a, nk=nk, Ts=self.Ts, cov=cov)

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
    def _lstsq_lstsq(Phi, y):
        theta, res, rank, s = scipy.linalg.lstsq(Phi, y)

        e = y - (Phi @ theta)
        var_e = np.var(e)
        cov = var_e * scipy.linalg.inv(Phi.T @ Phi)
        return theta, cov

    @staticmethod
    def _lstsq_pinv(Phi, y):
        theta = scipy.linalg.pinv(Phi) @ y

        e = y - (Phi @ theta)
        var_e = np.var(e)
        cov = var_e * scipy.linalg.inv(Phi.T @ Phi)
        return theta, cov

    @staticmethod
    def _lstsq_qr(Phi, y):
        Q, R = scipy.linalg.qr(Phi, mode="economic")
        theta = scipy.linalg.solve_triangular(R, Q.T @ y)

        e = y - (Phi @ theta)
        var_e = np.var(e)
        cov = var_e * scipy.linalg.inv(R.T @ R)
        return theta, cov

    @staticmethod
    def _lstsq_svd(Phi, y, l):
        U, s, Vh = scipy.linalg.svd(Phi, full_matrices=False)
        print(s)
        Sigma = np.diag(1 / s)

        if l > 0:
            rho = np.diag(s**2 / (s**2 + l))
            theta = Vh.T @ rho @ Sigma @ U.T @ y
        else:
            theta = Vh.T @ Sigma @ U.T @ y

        e = y - (Phi @ theta)
        var_e = np.var(e)
        if l > 0:
            Sigma_sqr = np.diag(s**2 / (s**2 + l) ** 2)
        else:
            Sigma_sqr = np.diag(1 / s**2)
        cov = var_e * (Vh @ Sigma_sqr @ Vh.T)
        return theta, cov

    @staticmethod
    def name():
        return "arx"
