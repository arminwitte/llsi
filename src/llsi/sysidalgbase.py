#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

import traceback as tb
from abc import ABC, abstractmethod, abstractstaticmethod

import numpy as np

from .sysidresults import SysIdResults


class SysIdAlgBase(ABC):
    def __init__(self, data, y_name, u_name, settings):
        self.y = data[y_name]
        self.u = data[u_name]
        self.Ts = data.Ts
        self.settings = settings

    def ident(self, order):
        try:
            mod = self._ident(order)
            y_hat = mod.simulate(self.u)
            residuals = mod.residuals(self.y, y_hat)
            success = True
            message = ""
        except Exception as e:
            traceback_str = "".join(tb.format_exception(None, e, e.__traceback__))
            mod = None
            residuals = None
            success = False
            message = traceback_str

        res = SysIdResults(
            mod=mod, residuals=residuals, success=success, message=message
        )

        return res

    @abstractmethod
    def _ident(self, order):
        pass

    @abstractstaticmethod
    def name():
        pass

    @staticmethod
    def _sse(y, y_hat):
        e = y - y_hat
        with np.errstate(over="ignore", invalid="ignore"):
            sse = e.T @ e
        return sse
