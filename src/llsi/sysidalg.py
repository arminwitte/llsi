#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

import numpy as np
import scipy.optimize

from .sysidalgbase import SysIdAlgBase


class PEM_Poly(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings={}):
        super().__init__(data, y_name, u_name, settings=settings)

    def ident(self, order):
        pass

    @staticmethod
    def name():
        return "pem_poly"


class PEM_SS(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings={}):
        super().__init__(data, y_name, u_name, settings=settings)
        alg = sysidalg.get_creator("po-moesp")
        # alg = sysidalg.get_creator('n4sid')
        self.alg_inst = alg(data, y_name, u_name)

    def ident(self, order):
        mod = self.alg_inst.ident(order)
        y_hat = mod.simulate(self.u)
        sse0 = self._sse(self.y, y_hat)

        def fun(x):
            mod.reshape(x)
            y_hat = mod.simulate(self.u)
            sse = self._sse(self.y, y_hat)
            # print("{:10.6g}".format(sse / sse0))
            return sse / sse0

        x0 = mod.vectorize()
        res = scipy.optimize.minimize(fun, x0)
        # res = scipy.optimize.minimize(fun,x0,method='nelder-mead')
        # res = scipy.optimize.minimize(fun,x0,method='BFGS')

        mod.reshape(res.x)

        J = scipy.optimize.approx_fprime(res.x, fun).reshape(1, -1)
        var_e = np.var(self.y - mod.simulate(self.u))
        mod.cov = var_e * (J.T @ J)

        print(J)

        return mod

    @staticmethod
    def name():
        return "pem_ss"


class SysIdAlgFactory:
    def __init__(self):
        self.creators = {}
        self.default_creator_name = None

    def register_creator(self, creator, default=False):
        name = creator.name()
        if default:
            self.default_creator_name = name
        self.creators[name] = creator

    def get_creator(self, name=None):
        if name:
            c = self.creators[name]
        else:
            c = self.creators[self.default_creator_name]
        return c


sysidalg = SysIdAlgFactory()

from .arx import ARX
from .subspace import N4SID, PO_MOESP

sysidalg.register_creator(N4SID)
sysidalg.register_creator(PO_MOESP, default=True)
sysidalg.register_creator(PEM_SS)
sysidalg.register_creator(ARX)


def sysid(data, y_name, u_name, order, method=None, settings={}):
    alg = sysidalg.get_creator(method)
    alg_inst = alg(data, y_name, u_name, settings=settings)
    mod = alg_inst.ident(order)
    return mod
