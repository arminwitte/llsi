#!/usr/bin/env python3
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

from abc import ABC, abstractmethod

import numpy as np


class SysIdAlgBase(ABC):
    def __init__(self, data, y_name, u_name, settings):
        self.input_names = u_name
        self.output_names = y_name

        y_name = np.atleast_1d(y_name)  # now I can iterate over
        y = []
        for name in y_name:
            y.append(data[name])
        self.y = np.atleast_2d(y).T

        u_name = np.atleast_1d(u_name)  # now I can iterate over
        u = []
        for name in u_name:
            u.append(data[name])
        self.u = np.atleast_2d(u).T

        self.Ts = data.Ts
        self.settings = settings

    @abstractmethod
    def ident(self, order):
        pass

    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    def _sse(y, y_hat):
        e = y - y_hat
        with np.errstate(over="ignore", invalid="ignore"):
            sse = e.T @ e
        return np.sum(sse)
