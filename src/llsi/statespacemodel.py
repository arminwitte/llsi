#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 22:54:53 2021

@author: armin
"""

import json

import numpy as np
import scipy.linalg

from .ltimodel import LTIModel


class StateSpaceModel(LTIModel):
    """
    State Space model class
    https://en.wikipedia.org/wiki/State-space_representation
    """

    def __init__(self, A=None, B=None, C=None, D=None, Ts=1.0, nx=0, nu=1, ny=1):
        """

        Parameters
        ----------
        A : TYPE, optional
            DESCRIPTION. The default is None.
        B : TYPE, optional
            DESCRIPTION. The default is None.
        C : TYPE, optional
            DESCRIPTION. The default is None.
        D : TYPE, optional
            DESCRIPTION. The default is None.
        Ts : TYPE, optional
            DESCRIPTION. The default is 1.0.
        Nx : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        super().__init__(Ts=Ts)

        # set A matrix and number of states
        if A is not None:
            self.A = np.array(A)
            self.nx = self.A.shape[0]
        else:
            self.nx = nx
            self.A = np.zeros((self.nx, self.nx))

        # set B matrix and number of inputs
        if B is not None:
            self.B = np.array(B).reshape(self.nx, -1)
            self.nu = self.B.shape[1]
        else:
            self.nu = nu
            self.B = np.zeros((self.nx, self.nu))

        # set C matrix and number of outputs
        if C is not None:
            self.C = np.array(C).reshape(-1, self.nx)
            self.ny = self.C.shape[0]
        else:
            self.ny = ny
            self.C = np.zeros((self.ny, self.nx))

        if D is not None:
            self.D = np.array(D).reshape(self.ny, self.nu)
        else:
            self.D = np.zeros((self.ny, self.nu))

        self.cov = None

    def vectorize(self):
        theta = np.vstack(
            [
                self.A.reshape(-1, 1),
                self.B.reshape(-1, 1),
                self.C.reshape(-1, 1),
                self.D.reshape(-1, 1),
            ]
        )

        return np.array(theta).ravel()

    def reshape(self, theta: np.ndarray):
        nx = self.nx
        nu = self.nu
        ny = self.ny

        na = nx * nx
        nb = nx * nu
        nc = ny * nx
        nd = ny * nu

        self.A = theta[:na].reshape(nx, nx)
        self.B = theta[na : na + nb].reshape(nx, nu)
        self.C = theta[na + nb : na + nb + nc].reshape(ny, nx)
        self.D = theta[na + nb + nc :].reshape(ny, nu)

    def simulate(self, u: np.ndarray):
        u = np.atleast_2d(u)
        u = u.reshape(self.nu,-1)
        N = u.shape[0]
        # TODO: initialize x properly
        x1 = np.zeros((self.nx, 1))
        y = np.empty((N, self.ny))
        # assert u.shape[1] == self.nu
        for i, u_ in enumerate(u.T):
            u_ = u_.T

            #u_ = u_.reshape(1, self.nu)
            x = x1
            with np.errstate(over="ignore", invalid="ignore"):
                x1 = self.A @ x + self.B @ u_
                y_ = self.C @ x + self.D @ u_

            # print(f"u:{u_}")
            # print(f"x:{x}")
            # print(f"x1:{x1}")
            # print(f"y:{y_}")

            y[i, :] = y_

        return y

    @classmethod
    def from_PT1(cls, K: float, tauC: float, Ts=1.0):
        t = 2 * tauC
        tt = 1 / (Ts + t)
        b = K * Ts * tt
        a = (Ts - t) * tt

        B = [(1 - a) * b]
        D = b

        A = [[-a]]
        C = [1]

        mod = cls(A=A, B=B, C=C, D=D, Ts=Ts, nx=1)

        return mod

    def to_ss(self, continuous=False, method="bilinear") -> scipy.signal.StateSpace:
        from scipy import signal

        if continuous:
            A, B, C, D = self._d2c(
                self.A, self.B, self.C, self.D, self.Ts, method=method
            )
            sys = signal.StateSpace(A, B, C, D)
        else:
            sys = signal.StateSpace(self.A, self.B, self.C, self.D, dt=self.Ts)
        return sys

    @staticmethod
    def _d2c(
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        Ts: float,
        method="bilinear",
    ):
        # https://math.stackexchange.com/questions/3820100/discrete-time-to-continuous-time-state-space
        if method in "bilinear":
            return StateSpaceModel._d2c_bilinear(A, B, C, D, Ts)
        else:
            return StateSpaceModel._d2c_euler(A, B, C, D, Ts)

    @staticmethod
    def _d2c_bilinear(
        A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, Ts: float
    ):
        I = np.eye(*A.shape)
        AI = scipy.linalg.inv(A + I)
        A_ = 2.0 / Ts * (A - I) @ AI
        B_ = 2.0 / Ts * (I - (A - I) @ AI) @ B
        C_ = C @ AI
        D_ = D - C @ AI @ B
        return A_, B_, C_, D_

    @staticmethod
    def _d2c_euler(
        A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, Ts: float
    ):
        A_ = (A - np.eye(*A.shape)) / Ts
        B_ = B / Ts
        C_ = C
        D_ = D
        return A_, B_, C_, D_

    def to_tf(self, continuous=False, method="bilinear"):
        sys = self.to_ss(continuous=continuous, method=method)
        return sys.to_tf()

    def to_zpk(self, continuous=False, method="bilinear"):
        sys = self.to_ss(continuous=continuous, method=method)
        return sys.to_zpk()

    def to_controllable_form(self):
        tf = self.to_tf()
        ss = tf.to_ss()
        return StateSpaceModel(A=ss.A, B=ss.B, C=ss.C, D=ss.D, Ts=self.Ts)

    def reduce_order(self, n: int):
        """
        Perform order reduction using balanced truncation

        Parameters
        ----------
        n : int
            New (reduced) model order.

        Returns
        -------
        TYPE
            DESCRIPTION.
        s : TYPE
            DESCRIPTION.

        References
        ----------
        Stanisławski R., Rydel M., Latawiec K.J.: Modeling of discrete-time fractional- order state space systems using the balanced truncation method, Journal of the Franklin Institute, vol. 354, no. 7, 2017, pp. 3008-3020. http://doi.org/10.1016/j.jfranklin.2017.02.003

        """
        A = self.A
        B = self.B
        C = self.C

        if n > A.shape[0]:
            raise ValueError(f"New model order has to be <= {A.shape[0]} but is {n}")

        # controllability gramian
        W_c = scipy.linalg.solve_discrete_lyapunov(A, B @ B.T)

        # observability gramian
        W_o = scipy.linalg.solve_discrete_lyapunov(A.T, C.T @ C)

        # controllability matrix
        S = scipy.linalg.cholesky(W_c)

        # observability matrix
        R = scipy.linalg.cholesky(W_o)

        U, s, V = scipy.linalg.svd(S @ R.T)

        # truncation
        U1 = U[:, :n]
        s1 = s[:n]
        V1 = V[:, :n]
        Sigma1 = np.diag(1 / s1)

        # # Square root algorithm
        # # create transformation matrices
        # T_l = np.sqrt(Sigma1) @ U1.T @ R
        # T_r = S.T @ V1 @ np.sqrt(Sigma1)

        # balancing-free square root algorithm
        W, X = scipy.linalg.qr(S.T @ U1, mode="economic")
        Z, Y = scipy.linalg.qr(R.T @ V1, mode="economic")
        UE, sE, VE = scipy.linalg.svd(Z.T @ W)
        SigmaE = np.diag(1 / sE)
        T_l = np.sqrt(SigmaE) @ UE.T @ Z.T
        T_r = W @ VE @ np.sqrt(SigmaE)

        # apply transformation
        A_ = T_l @ A @ T_r
        B_ = T_l @ B
        C_ = C @ T_r

        return StateSpaceModel(A=A_, B=B_, C=C_, D=self.D, Ts=self.Ts), s

    @classmethod
    def from_scipy(cls, mod):
        ss = mod.ss()
        mod_out = cls(A=ss.A, B=ss.B, C=ss.C, D=ss.D, Ts=ss.dt)
        return mod_out

    @classmethod
    def from_fir(cls, mod):
        nk = mod.nk
        b = np.vstack([np.zeros((nk, 1)), mod.b.reshape(-1, 1)])
        n = b.ravel().shape[0] - 1
        A = np.diag(np.ones((n - 1,)), k=-1)
        B = np.zeros((n, 1))
        B[0] = 1.0
        C = b[1:].reshape(1, -1)
        D = b[0]
        mod_out = cls(A=A, B=B, C=C, D=D, Ts=mod.Ts)
        return mod_out

    def to_json(self, filename=None):
        data = {}
        data["A"] = self.A.tolist()
        data["B"] = self.B.tolist()
        data["C"] = self.C.tolist()
        data["D"] = self.D.tolist()
        data["Ts"] = self.Ts
        try:
            data["info"] = self.info.__repr__()
        except:
            data["info"] = {}
        data["nx"] = self.ny
        data["nu"] = self.ny
        data["ny"] = self.ny

        if filename is not None:
            with open(filename, "w") as f:
                json.dump(data, f)
                return

        return json.dumps(data)

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        mod = StateSpaceModel(
            A=data["A"],
            B=data["B"],
            C=data["C"],
            D=data["D"],
            Ts=data["Ts"],
            nx=data["nx"],
            nu=data["nu"],
            ny=data["ny"],
        )
        mod.info = data["info"]
        return mod

    def __repr__(self) -> str:
        s = f"StateSpaceModel with Ts={self.Ts}\n"
        s += f"A:\n{self.A}\n"
        s += f"B:\n{self.B}\n"
        s += f"C:\n{self.C}\n"
        s += f"D:\n{self.D}\n"
        return s

    def __str__(self):
        return self.__repr__()
