#!/usr/bin/env python3
"""
Created on Wed Aug  3 10:16:09 2022

@author: armin
"""

import numpy as np
import scipy.signal

from .ltimodel import LTIModel


class PolynomialModel(LTIModel):
    def __init__(
        self,
        a=None,
        b=None,
        na=1,
        nb=1,
        nu=1,
        ny=1,
        nk=0,
        cov=None,
        Ts=1.0,
        input_names=None,
        output_names=None,
    ):
        if input_names is None:
            input_names = []
        if output_names is None:
            output_names = []
        super().__init__(Ts=Ts, input_names=input_names, output_names=output_names)

        if a is not None:
            self.a = np.atleast_2d(a).T
            self.na = self.a.shape[0]
            self.ny = self.a.shape[1]
        else:
            self.na = na
            self.ny = ny
            self.a = np.ones((self.na, self.ny))

        if b is not None:
            self.b = np.atleast_2d(b).T
            self.nb = self.b.shape[0]
            self.nu = self.b.shape[1]
        else:
            self.nb = nb
            self.nu = nu
            self.b = np.ones((self.nb, self.nu))

        if self.ny > 1:
            raise ValueError("System seems to have multiple outputs. This is not implemented.")

        if self.nu > 1:
            raise ValueError("System seems to have multiple inputs. This is not implemented.")

        # norm
        if self.a.shape[0] > 0:
            self.b = self.b.ravel() / self.a[0, 0]
            self.a = self.a.ravel() / self.a[0, 0]

        self.nk = nk

        self.cov = cov

    def simulate(self, u):
        u = np.atleast_2d(u).ravel()
        N = u.shape[0]
        y = np.zeros((N, self.ny))
        a = self.a
        b = self.b
        na = self.na
        nb = self.nb
        nk = self.nk
        n = max(na, nb + nk)

        # init with for-loops
        for i in range(n):
            for j in range(i + 1):
                if i - j - nk >= 0 and j < nb:
                    y[i] += b[j] * u[i - j - nk]
            for j in range(1, i + 1):
                if i - j >= 0 and j < na:
                    with np.errstate(over="ignore", invalid="ignore"):
                        y[i] -= a[j] * y[i - j]

        # vectorize for speed
        for i in range(n, N):
            with np.errstate(over="ignore", invalid="ignore"):
                y[i] += b.T @ u[i - nk : i - nb - nk : -1]
                y[i] -= a.T[1:] @ y[i - 1 : i - 1 - (na - 1) : -1]
        return y

    def frequency_response(self, omega=np.logspace(-3, 2)):
        a = self.a
        b = self.b

        z = np.exp(1j * omega * self.Ts)
        H = []

        for z_ in z:
            za = np.power(z_, -np.arange(len(a)))
            zb = np.power(z_, -np.arange(len(b)))
            h = (b @ zb) / (a @ za)
            H.append(h)

        return omega, np.array(H)

    def vectorize(self):
        return np.hstack((self.b, self.a[1:])).ravel()

    def reshape(self, theta):
        # ensure 1d array
        theta = np.array(theta).ravel()
        self.b = theta[: self.nb * self.nu]
        self.a = np.hstack(([1.0], theta[self.nb * self.nu :]))

    def to_tf(self):
        return scipy.signal.TransferFunction(self.b, self.a, dt=self.Ts)

    @classmethod
    def from_scipy(cls, mod):
        tf = mod.tf()
        mod_out = cls(a=tf.den, b=tf.num, Ts=tf.dt)
        return mod_out

    def __repr__(self):
        s = f"PolynomialModel with Ts={self.Ts}\n"
        s += f"input(s): {self.input_names}\n"
        s += f"output(s): {self.output_names}\n"
        s += f"b:\n{self.b}\n"
        s += f"a:\n{self.a}\n"

        return s

    def __str__(self):
        return self.__repr__()
