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
    def __init__(self, **kwargs):
        super().__init__(Ts=kwargs.get("Ts", 1.0))
        self.a = kwargs.get("a", np.array([]))
        self.b = kwargs.get("b", np.array([]))
        if len(self.a) > 0:
            self.b = self.b / self.a[0]
            self.a = self.a / self.a[0]
        self.na = self.a.ravel().shape[0]
        self.nb = self.b.ravel().shape[0]
        self.nk = kwargs.get("nk", 0)

        self.cov = kwargs.get("cov")

    def simulate(self, u):
        u = u.ravel()
        N = u.shape[0]
        y = np.zeros_like(u)
        # a = self.a.reshape(-1,1)
        # b = self.b.reshape(-1,1)
        a = self.a.ravel()
        b = self.b.ravel()
        na = a.shape[0]  #!
        nb = b.shape[0]  #!
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
                    y[i] -= a[j] * y[i - j]

        # vectorize for speed
        for i in range(n, N):
            y[i] += b.T @ u[i - nk : i - nb - nk : -1]
            y[i] -= a.T[1:] @ y[i - 1 : i - 1 - (na - 1) : -1]
        return y

    def vectorize(self):
        self.na = self.a.ravel().shape[0]
        self.nb = self.b.ravel().shape[0]
        return np.hstack((self.b, self.a[1:])).ravel()

    def reshape(self, theta):
        self.b = theta[: self.nb]
        self.a = np.hstack(([1.0], theta[self.nb :]))

    def to_tf(self):
        return scipy.signal.TransferFunction(self.b, self.a, dt=self.Ts)

    def __repr__(self):
        s = f"b:\n{self.b}\n"
        s += f"a:\n{self.a}\n"
        return s

    def __str__(self):
        return self.__repr__()
