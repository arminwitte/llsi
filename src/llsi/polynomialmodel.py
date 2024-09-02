#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:16:09 2022

@author: armin
"""
import numpy as np
import scipy.signal

from .ltimodel import LTIModel


class PolynomialModel(LTIModel):
    def __init__(self, a=None, b=None, na=1, nb=1, nu=1, ny=1, nk=0, cov=None, Ts=1.0):
        super().__init__(Ts=Ts)

        if a is not None:
            self.a = np.atleast_2d(a)
            self.na = self.a.shape[0]
            self.ny = self.a.shape[1]
        else:
            self.na = na
            self.ny = ny
            self.a = np.ones((self.na, self.ny))

        if b is not None:
            self.b = np.atleast_2d(b)
            self.nb = self.b.shape[0]
            self.nu = self.b.shape[1]
        else:
            self.nb = nb
            self.nu = nu
            self.b = np.ones((self.nb, self.nu))

        # norm
        if self.a.shape[0] > 0:
            self.b = self.b / self.a[0, 0]
            self.a = self.a / self.a[0, 0]

        self.nk = nk

        self.cov = cov

    def simulate(self, u):
        N = u.shape[0]
        y = np.zeros((N, self.ny))
        a = self.a
        b = self.b
        na = self.na
        nb = self.nb
        nk = self.nk
        n = max(na, nb + nk)

        # print(a.T[1:].shape)

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

    def vectorize(self):
        return np.hstack((self.b.ravel(), self.a[1:].ravel())).ravel()

    def reshape(self, theta):
        self.b = theta[: self.nb * self.nu].reshape(self.nb, self.nu)
        self.a = np.hstack(([1.0], theta[self.nb * self.nu :])).reshape(
            self.na, self.ny
        )

    def to_tf(self):
        return scipy.signal.TransferFunction(self.b, self.a, dt=self.Ts)

    @classmethod
    def from_scipy(cls, mod):
        tf = mod.tf()
        mod_out = cls(a=tf.den, b=tf.num, Ts=tf.dt)
        return mod_out

    def __repr__(self):
        s = f"PolynomialModel with Ts={self.Ts}"
        s += f"b:\n{self.b}\n"
        s += f"a:\n{self.a}\n"

        return s

    def __str__(self):
        return self.__repr__()
