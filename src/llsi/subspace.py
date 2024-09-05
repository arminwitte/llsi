#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:54:55 2021

@author: armin
"""

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

        # for x_ in x.T:
        #     x_ = x_.ravel().T
        #     for i in range(n):
        #         A.append(x_[i : -n + i])

        return np.array(A)

    def _abcd_state(self, Xf, s, n, r):
        Y_ = np.vstack((Xf[:, 1 : s - 1], self.y[r : r + s - 2, :].T))
        X_ = np.vstack((Xf[:, : s - 2], self.u[r : r + s - 2, :].T))

        # Theta = Y_ @ np.linalg.pinv(X_)

        ###########################
        lmbd = self.settings.get("lambda", 0.0)
        # print(l)
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
        # print(A)

        # ====================================================================

        print(f"U2.T shape: {U2.T.shape}")
        P = np.split(U2.T, U2.T.shape[1] // ny, axis=1)
        print(f"Pi shape: {P[0].shape}")

        nn = len(P) * P[0].shape[0]
        print(f"nn: {nn}")
        rny = r
        print(f"rny: {rny}")
        r = r // ny
        print(f"r: {r}")
        print(f"n: {n}")
        A_ = np.zeros((nn, ny + n))
        print(f"A_ shape: {A_.shape}")
        P1 = np.vstack(P)
        rr = P[0].shape[0]
        # for i, Pi in enumerate(P):
        #     A_[i * rr : (i + 1) * rr, :ny] = Pi
        A_[:, :ny] = P1

        for i in range(1, r):
            print(i)
            Pi_tilda = np.hstack(P[i:])
            print(f"Pi_tilda shape: {Pi_tilda.shape}")
            Ori = Or[: rny - (ny * i)]
            print(f"Ori shape: {Ori.shape}")
            N = Pi_tilda @ Ori
            print(f"N shape: {N.shape}")
            A_[(i - 1) * rr : i * rr, ny:] = N

        # P = U2.T.reshape(-1,ny,order="F")
        # # print(P)
        # # A1_ = P.ravel(order="F")
        # A1_ = P

        # nn = A1_.shape[0]
        # A_ = np.zeros((nn, ny + n))
        # A_[:, 0:ny] = A1_.reshape(-1, ny)

        # for i in range(1, r):
        #     Pi = P[:, i:r]
        #     print(f"Pi shape: {Pi.shape}")
        #     Oi = Or[0 : r - i, :]
        #     print(f"Oi shape: {Oi.shape}")
        #     Ni = Pi @ Oi
        #     j = (i - 1) * (r - n)
        #     A_[j : j + Ni.shape[0], ny : ny + Ni.shape[1]] = Ni

        # print(A_)

        # M = (U2.T @ L31 @ np.linalg.inv(L11)).reshape(-1, ny, order="F")
        M = U2.T @ L31 @ np.linalg.inv(L11)
        Mi = np.split(M, M.shape[1] // nu, axis=1)
        M = np.vstack(Mi)
        print(f"M shape: {M.shape}")

        x_, *_ = scipy.linalg.lstsq(A_, M)
        # x_ = scipy.linalg.pinv(A_) @ M

        D = x_[:ny, :]
        B = x_[ny:, :]

        return A, B, C, D

    @staticmethod
    def lq(A):
        Q, R = scipy.linalg.qr(A.T, mode="economic")
        return R.T, Q.T


class N4SID(SubspaceIdent):
    def __init__(self, data, y_name, u_name, settings={}):
        super().__init__(data, y_name, u_name, settings=settings)

        # estimate extended observability matrix and states. Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)

    def ident(self, order):
        if isinstance(order, (tuple, list)):
            n = order[0]
        else:
            n = order

        r = (2 * n + 1) * self.nu * self.ny  # window length

        Y = self.hankel(self.y, 2 * r)
        U = self.hankel(self.u, 2 * r)

        s = Y.shape[1]

        Yp = Y[0:r, :]
        Up = U[0:r, :]

        Yf = Y[r : 2 * r, :]
        Uf = U[r : 2 * r, :]

        Wp = np.vstack((Up, Yp))
        # Wf = np.vstack((Uf,Yf))
        Psi = np.vstack((np.vstack((Uf, Wp)), Yf))

        L, Q = self.lq(Psi)

        L11 = L[0:r, 0:r]
        # L12 = L[  0:r  ,r:2*r]
        # L21 = L[  r:3*r,0:r]
        L22 = L[r : 3 * r, r : 3 * r]
        L31 = L[3 * r : 4 * r, 0:r]
        L32 = L[3 * r : 4 * r, r : 3 * r]

        # print(L32.shape, np.linalg.pinv(L22).shape, Wp.shape)
        Gamma_r = L32 @ np.linalg.pinv(L22) @ Wp  # oblique projection
        # print(Gamma_r.shape)
        U_, s_, V_ = scipy.linalg.svd(Gamma_r, full_matrices=False)
        # print(U_.shape, s_.shape, V_.shape)

        self.singular_values = s_

        U1 = U_[:, 0:n]
        U2 = U_[:, n:r]
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))
        # Sigma_sqrt = np.zeros(Gamma_r.shape, s_.dtype)
        # np.fill_diagonal(Sigma_sqrt, np.sqrt(s_))
        # Sigma_sqrt = Sigma_sqrt[:n, :]
        # Sigma_sqrt = scipy.linalg.diagsvd(s_, *Gamma_r.shape)
        V1 = V_[0:n, :]
        # V1 = V_[:, :n]
        # V2 = V_[:,n:r]

        # Or = U1 @ Sigma_sqrt  # extended observability matrix
        Xf = (
            Sigma_sqrt @ V1
        )  # state matrix # TANGIRALA SAYS IT SHOULD BE TRANSPOSED !?!

        # print(s,n,r)
        A, B, C, D = self._abcd_state(Xf, s, n, r)
        # A, B, C, D = self._abcd_observability_matrix(self, U1, U2, L11, L31, Sigma_sqrt, n, r):

        mod = StateSpaceModel(A=A, B=B, C=C, D=D, Ts=self.Ts)
        mod.info["Hankel singular values"] = s_

        return mod

    @staticmethod
    def name():
        return "n4sid"


# class N4SID2(SubspaceIdent):
#     def __init__(self, data, y_name, u_name, settings={}):
#         super().__init__(data, y_name, u_name, settings=settings)

#         # estimate extended observability matrix and states. Then estimate A, B, C, and D in one go.
#         # (Tangirala 2014)

#     def ident(self, order):
#         if isinstance(order, (tuple, list)):
#             n = order[0]
#         else:
#             n = order

#         r = 2 * n + 1  # window length

#         Y = self.hankel(self.y, 2 * r)
#         U = self.hankel(self.u, 2 * r)

#         s = Y.shape[1]

#         Yp = Y[0:r, :]
#         YpPlus = Y[0:r+1, :]
#         Up = U[0:r, :]
#         UpPlus = U[0:r+1, :]

#         Yf = Y[r : 2 * r, :]
#         YfMinus = Y[r+1 : 2 * r, :]
#         Uf = U[r : 2 * r, :]
#         UfMinus = U[r+1 : 2 * r, :]

#         Wp = np.vstack((Up, Yp))
#         WpPlus = np.vstack((UpPlus, YpPlus))
#         # Wf = np.vstack((Uf,Yf))


#         # Projection 1
#         Psi = np.vstack((np.vstack((Uf, Wp)), Yf))

#         L, _ = self.lq(Psi)

#         L22 = L[r : 3 * r, r : 3 * r]
#         L32 = L[3 * r : 4 * r, r : 3 * r]


#         # Projection 2
#         PsiAst = np.vstack((np.vstack((Uf, Wp)), Yf))

#         L, _ = self.lq(PsiAst)

#         LAst22 = L[r-1 : 3 * r+1, r-1 : 3 * r+1]
#         LAst32 = L[3 * r+1 : 4 * r, r-1 : 3 * r+1]


#         # print(L32.shape, np.linalg.pinv(L22).shape, Wp.shape)
#         Gamma_r = L32 @ np.linalg.pinv(L22) @ Wp  # oblique projection
#         GammaAst_r = LAst32 @ np.linalg.pinv(LAst22) @ WpPlus  # oblique projection
#         U_, s_, _ = scipy.linalg.svd(Gamma_r, full_matrices=False)

#         self.singular_values = s_

#         U1 = U_[:,0:n]
#         Sigma_sqrt = np.diag(np.sqrt(s_[:n]))

#         Or = U1 @ Sigma_sqrt
#         Xf = np.linalg.pinv(Or) @ Gamma_r
#         XfPlus =  np.linalg.pinv(Or[:-1,:]) @ GammaAst_r

#         Y_ = np.vstack((XfPlus[:, : s - 2], self.y[r : r + s - 2].T))
#         X_ = np.vstack((Xf[:, : s - 2], self.u[r : r + s - 2].T))
#         Theta = Y_ @ np.linalg.pinv(X_)
#         A = Theta[:n, :n]
#         B = Theta[:n, n].ravel()
#         C = Theta[n, :n].ravel()
#         D = Theta[n, n]

#         mod = StateSpaceModel(A=A, B=B, C=C, D=D, Ts=self.Ts)
#         mod.info["Hankel singular values"] = s_

#         return mod

#     @staticmethod
#     def name():
#         return "n4sid2"


class PO_MOESP(SubspaceIdent):
    def __init__(self, data, y_name, u_name, settings={}):
        super().__init__(data, y_name, u_name, settings=settings)

        # estimate extended observability matrix and states. Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)

    def ident(self, order):
        # Tangirala 2014
        # Algorithm 23.3
        if isinstance(order, (tuple, list)):
            n = order[0]
        else:
            n = order

        # N = self.y.shape[0]

        r = (2 * n + 1) * self.nu * self.ny  # window length

        Y = self.hankel(self.y, 2 * r)
        U = self.hankel(self.u, 2 * r)

        # s = Y.shape[1]

        Yp = Y[0:r, :]
        Up = U[0:r, :]

        Yf = Y[r : 2 * r, :]
        Uf = U[r : 2 * r, :]

        Wp = np.vstack((Up, Yp))
        # Wf = np.vstack((Uf,Yf))
        # Psi = 1.0 / N * np.vstack((np.vstack((Uf, Wp)), Yf))
        Psi = np.vstack((np.vstack((Uf, Wp)), Yf))

        L, Q = self.lq(Psi)

        L11 = L[0:r, 0:r]
        # L12 = L[  0:r  ,r:3*r]
        # L21 = L[  r:3*r,0:r]
        # L22 = L[  r:3*r,r:3*r]
        L31 = L[3 * r : 4 * r, 0:r]
        L32 = L[3 * r : 4 * r, r : 3 * r]

        # Gamma_r = 1./np.sqrt(N) * L32
        Gamma_r = L32
        # print(Gamma_r.shape)
        U_, s_, V_ = scipy.linalg.svd(Gamma_r, full_matrices=False)

        self.singular_values = s_

        # print(s_)

        U1 = U_[:, 0:n]
        U2 = U_[:, n:r]
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))
        # V1 = V_[:n, :]
        # V1 = V_[:,0:n]
        # V2 = V_[:,n:r]

        # Or = U1 @ Sigma_sqrt  # extended observability matrix
        # Xf = np.linalg.pinv(Or) @ Gamma_r
        # print(Xf.shape)

        # C = Or[0, :]# TODO: might be wrong!!!
        # A = scipy.linalg.pinv(Or[0:-1, :]) @ Or[1:, :]
        # # A = scipy.linalg.lstsq(Or[0:-1,:],Or[1:,:])
        # # print(A)

        # P = U2.T
        # # print(P)
        # A1_ = P.ravel(order="F")

        # nn = A1_.shape[0]
        # ny = 1
        # A_ = np.zeros((nn, ny + n))
        # A_[:, 0:ny] = A1_.reshape(-1, ny)

        # for i in range(1, r):
        #     Pi = P[:, i:r]
        #     Oi = Or[0 : r - i, :]
        #     Ni = Pi @ Oi
        #     j = (i - 1) * (r - n)
        #     A_[j : j + Ni.shape[0], ny : ny + Ni.shape[1]] = Ni

        # # print(A_)

        # M = (U2.T @ L31 @ np.linalg.inv(L11)).ravel(order="F")

        # x_, *_ = scipy.linalg.lstsq(A_, M)
        # # x_ = scipy.linalg.pinv(A_) @ M

        # D = x_[0:ny]
        # B = x_[ny : ny + n]

        # print(s,n,r)
        A, B, C, D = self._abcd_observability_matrix(U1, U2, L11, L31, Sigma_sqrt, n, r)
        # A, B, C, D = self._abcd_state(Xf, s, n, r)

        mod = StateSpaceModel(A=A, B=B, C=C, D=D, Ts=self.Ts)
        mod.info["Hankel singular values"] = s_

        return mod

    @staticmethod
    def name():
        return "po-moesp"
