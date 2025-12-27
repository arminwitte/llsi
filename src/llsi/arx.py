#!/usr/bin/env python3
"""
Created on Fri Aug 12 01:34:47 2022

@author: armin
"""

import logging

import numpy as np
import scipy.linalg

from .polynomialmodel import PolynomialModel
from .sysidalgbase import SysIdAlgBase


class ARX(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)
        if self.u.shape[1] > 1:
            raise ValueError("There seem to be multiple inputs. This is not implemented.")

        if self.y.shape[1] > 1:
            raise ValueError("There seem to be multiple outputs. This is not implemented.")

        self.u = self.u.ravel()
        self.y = self.y.ravel()
        self.logger = logging.getLogger(__name__)

    def ident(self, order):
        na, nb, nk = order
        Phi, y = self._observations(na, nb, nk)

        # theta = scipy.linalg.pinv(Phi) @ y
        # theta, res, rank, s = scipy.linalg.lstsq(Phi, y)

        lstsq_method = self.settings.get("lstsq_method", "qr")
        lmb = self.settings.get("lambda", 0.0)
        if lstsq_method in "pinv":
            theta, cov = self._lstsq_pinv(Phi, y)
        elif lstsq_method in "lstsq":
            theta, cov = self._lstsq_lstsq(Phi, y)
        elif lstsq_method in "qr":
            theta, cov = self._lstsq_qr(Phi, y, lmb)
        elif lstsq_method in "svd":
            theta, cov = self._lstsq_svd(Phi, y, lmb)
        self.logger.debug(f"theta:\n{theta}")

        b = theta[:nb]
        a = np.hstack(([1.0], theta[nb:]))

        mod = PolynomialModel(
            b=b,
            a=a,
            nk=nk,
            Ts=self.Ts,
            cov=cov,
            input_names=self.input_names,
            output_names=self.output_names,
        )

        return mod

    def _observations(self, na, nb, nk):
        u = self.u.ravel()
        y = self.y.ravel()
        nn = max(nb + nk, na)
        N = u.ravel().shape[0]
        Phi = np.empty((N - nn, nb + na))
        for i in range(nb):
            Phi[:, i] = u[nn - i - nk : N - i - nk]
        for i in range(na):
            Phi[:, nb + i] = -y[nn - i - 1 : N - i - 1]

        y_ = y[nn:N]

        return Phi, y_

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
    def _lstsq_qr(Phi, y, lmb):
        Phi_ = np.vstack([Phi, lmb * np.eye(Phi.shape[1])])
        y_ = np.vstack([y.reshape(-1, 1), np.zeros((Phi.shape[1], 1))]).ravel()
        Q, R = scipy.linalg.qr(Phi_, mode="economic")
        theta = scipy.linalg.solve_triangular(R, Q.T @ y_)

        e = y - (Phi @ theta)
        var_e = np.var(e)
        cov = var_e * scipy.linalg.inv(R.T @ R)
        return theta, cov

    @staticmethod
    def _lstsq_svd(Phi, y, lmb):
        U, s, Vh = scipy.linalg.svd(Phi, full_matrices=False)
        Sigma = np.diag(1 / s)

        if lmb > 0:
            rho = np.diag(s**2 / (s**2 + lmb))
            theta = Vh.T @ rho @ Sigma @ U.T @ y
        else:
            theta = Vh.T @ Sigma @ U.T @ y

        e = y - (Phi @ theta)
        var_e = np.var(e)
        if lmb > 0:
            Sigma_sqr = np.diag(s**2 / (s**2 + lmb) ** 2)
        else:
            Sigma_sqr = np.diag(1 / s**2)
        cov = var_e * (Vh @ Sigma_sqr @ Vh.T)
        return theta, cov

    @staticmethod
    def name():
        return "arx"


class FIR(ARX):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)

    def ident(self, order):
        order = list(order)
        order[0] = 0
        return super().ident(order)

    @staticmethod
    def name():
        return "fir"
