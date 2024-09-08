#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

import numpy as np
import scipy.optimize

from .sysidalg import sysidalg
from .sysidalgbase import SysIdAlgBase


class PEM(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings={}):
        super().__init__(data, y_name, u_name, settings=settings)
        init = self.settings.get("init", "arx")
        alg = sysidalg.get_creator(init)
        # alg = sysidalg.get_creator('n4sid')
        self.alg_inst = alg(data, y_name, u_name)

    def ident(self, order):
        mod = self.alg_inst.ident(order)
        y_hat = mod.simulate(self.u)
        sse0 = self._sse(self.y, y_hat)
        lambda_l1 = self.settings.get("lambda_l1", 0.0)
        lambda_l2 = self.settings.get("lambda_l2", 0.0)

        def fun(x):
            mod.reshape(x)
            y_hat = mod.simulate(self.u)
            sse = self._sse(self.y, y_hat)
            sse = np.nan_to_num(sse, nan=1e300)
            # print("{:10.6g}".format(sse / sse0))
            # return sse / sse0
            x_ = x.ravel()
            J = sse + lambda_l1 * np.sum(np.abs(x)) + lambda_l2 * x_.T @ x_
            print("{:10.6g}".format(J))
            return J

        x0 = mod.vectorize()
        method = self.settings.get("minimizer", "nelder-mead")
        # res = scipy.optimize.minimize(
           # fun, x0, method=method, options={"maxiter": 200, "maxfev": 200}
        # (
        # res = scipy.optimize.basinhopping(fun, x0, niter=1, minimizer_kwargs={"method":"BFGS", "options":{"maxiter":20}}, disp=True)
        res = scipy.optimize.minimize(fun,x0,method='nelder-mead')
        res = scipy.optimize.minimize(fun,res.x,method='BFGS',options={"gtol":1e-3})

        mod.reshape(res.x)

        J = scipy.optimize.approx_fprime(res.x, fun).reshape(1, -1)
        var_e = np.var(self.y - mod.simulate(self.u))
        mod.cov = var_e * (J.T @ J)

        return mod

    @staticmethod
    def name():
        return "pem"


######################################################################################
# CONVENIENCE CLASSES
######################################################################################


class OE(PEM):
    def __init__(self, data, y_name, u_name, settings={}):
        settings["init"] = "arx"
        super().__init__(data, y_name, u_name, settings=settings)

    @staticmethod
    def name():
        return "oe"
