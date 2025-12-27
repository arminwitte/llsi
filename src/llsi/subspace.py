#!/usr/bin/env python3
"""
Created on Sun Apr  4 21:54:55 2021

@author: armin
"""

import logging

import numpy as np
import scipy.linalg

from .statespacemodel import StateSpaceModel
from .sysidalgbase import SysIdAlgBase


class SubspaceIdent(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings):
        super().__init__(data, y_name, u_name, settings=settings)
        self.nu = self.u.shape[1]
        self.ny = self.y.shape[1]

    @staticmethod
    def hankel(x, n):
        A = []
        n = n // x.shape[1]
        for i in range(n):
            for x_ in x.T:
                x_ = x_.ravel().T
                A.append(x_[i : -n + i])

        return np.array(A)

    def _abcd_state(self, Xf, s, n, r):
        Y_ = np.vstack((Xf[:, 1 : s - 1], self.y[r : r + s - 2, :].T))
        X_ = np.vstack((Xf[:, : s - 2], self.u[r : r + s - 2, :].T))

        # Theta = Y_ @ np.linalg.pinv(X_)

        ###########################
        lmbd = self.settings.get("lambda", 0.0)
        U, s, Vh = scipy.linalg.svd(X_, full_matrices=False)
        Sigma = np.diag(1 / s)

        rho = np.diag(s**2 / (s**2 + lmbd))
        Theta = Y_ @ (Vh.T @ rho @ Sigma @ U.T)
        ###########################

        A = Theta[:n, :n]
        B = Theta[:n, n:]
        C = Theta[n:, :n]
        D = Theta[n:, n:]
        return A, B, C, D

    def _abcd_observability_matrix(self, U1, U2, L11, L31, Sigma_sqrt, n, r):
        ny = self.ny
        nu = self.nu

        Or = U1 @ Sigma_sqrt  # extended observability matrix

        C = Or[:ny, :]  # TODO: might be wrong!!!
        A = scipy.linalg.pinv(Or[0:-ny, :]) @ Or[ny:, :]
        # A = scipy.linalg.lstsq(Or[0:-1,:],Or[1:,:])

        # ====================================================================

        P = np.split(U2.T, U2.T.shape[1] // ny, axis=1)

        nn = len(P) * P[0].shape[0]
        rny = r
        r = r // ny
        A_ = np.zeros((nn, ny + n))
        P1 = np.vstack(P)
        rr = P[0].shape[0]
        A_[:, :ny] = P1

        for i in range(1, r):
            Pi_tilda = np.hstack(P[i:])
            Ori = Or[: rny - (ny * i)]
            N = Pi_tilda @ Ori
            A_[(i - 1) * rr : i * rr, ny:] = N

        M = U2.T @ L31 @ np.linalg.inv(L11)
        Mi = np.split(M, M.shape[1] // nu, axis=1)
        M = np.vstack(Mi)

        x_, *_ = scipy.linalg.lstsq(A_, M)

        D = x_[:ny, :]
        B = x_[ny:, :]

        return A, B, C, D

    @staticmethod
    def lq(A):
        Q, R = scipy.linalg.qr(A.T, mode="economic")
        return R.T, Q.T


class N4SID(SubspaceIdent):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)
        self.logger = logging.getLogger(__name__)

        # estimate extended observability matrix and states.
        # Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)

        if self.y.shape[1] > 1 or self.u.shape[1] > 1:
            raise NotImplementedError("n4sid not implemented for multiple inputs or outputs.")

    def ident(self, order):
        if isinstance(order, (tuple, list)):
            n = order[0]
        else:
            n = order

        r = (2 * n + 1) * self.nu * self.ny  # window length

        Y = self.hankel(self.y, 2 * r)
        U = self.hankel(self.u, 2 * r)

        Yp = Y[0:r, :]
        Up = U[0:r, :]

        Yf = Y[r : 2 * r, :]
        Uf = U[r : 2 * r, :]

        Wp = np.vstack((Up, Yp))
        Psi = np.vstack((np.vstack((Uf, Wp)), Yf))

        L, Q = self.lq(Psi)
        L22 = L[r : 3 * r, r : 3 * r]
        L32 = L[3 * r : 4 * r, r : 3 * r]

        Gamma_r = L32 @ np.linalg.pinv(L22) @ Wp  # oblique projection

        U_, s_, V_ = scipy.linalg.svd(Gamma_r, full_matrices=False)

        self.singular_values = s_
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))

        # ====================================================================
        s = Y.shape[1]
        V1 = V_[0:n, :]
        Xf = Sigma_sqrt @ V1  # state matrix # TANGIRALA SAYS IT SHOULD BE TRANSPOSED !?!

        A, B, C, D = self._abcd_state(Xf, s, n, r)
        # ====================================================================
        # U1 = U_[:, 0:n]
        # U2 = U_[:, n:r]
        # L11 = L[0:r, 0:r]
        # L31 = L[3 * r : 4 * r, 0:r]

        # A, B, C, D = self._abcd_observability_matrix(U1, U2, L11, L31, Sigma_sqrt, n, r)
        # ====================================================================

        mod = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            Ts=self.Ts,
            input_names=self.input_names,
            output_names=self.output_names,
        )
        mod.info["Hankel singular values"] = s_

        return mod

    @staticmethod
    def name():
        return "n4sid"


class PO_MOESP(SubspaceIdent):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)
        self.logger = logging.getLogger(__name__)

        # estimate extended observability matrix and states.
        # Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)

    def ident(self, order):
        # Tangirala 2014
        # Algorithm 23.3
        if isinstance(order, (tuple, list)):
            n = order[0]
        else:
            n = order

        r = (2 * n + 1) * self.nu * self.ny  # window length

        Y = self.hankel(self.y, 2 * r)
        U = self.hankel(self.u, 2 * r)

        Yp = Y[0:r, :]
        Up = U[0:r, :]

        Yf = Y[r : 2 * r, :]
        Uf = U[r : 2 * r, :]

        Wp = np.vstack((Up, Yp))
        Psi = np.vstack((np.vstack((Uf, Wp)), Yf))

        L, Q = self.lq(Psi)

        L11 = L[0:r, 0:r]
        L31 = L[3 * r : 4 * r, 0:r]
        L32 = L[3 * r : 4 * r, r : 3 * r]

        Gamma_r = L32

        U_, s_, V_ = scipy.linalg.svd(Gamma_r, full_matrices=False)

        self.singular_values = s_

        self.logger.debug(f"s: {s_}")

        U1 = U_[:, 0:n]
        U2 = U_[:, n:r]
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))

        A, B, C, D = self._abcd_observability_matrix(U1, U2, L11, L31, Sigma_sqrt, n, r)

        mod = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            Ts=self.Ts,
            input_names=self.input_names,
            output_names=self.output_names,
        )
        mod.info["Hankel singular values"] = s_

        return mod

    @staticmethod
    def name():
        return "po-moesp"
