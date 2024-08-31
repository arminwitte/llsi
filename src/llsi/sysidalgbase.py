#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

from abc import ABC, abstractmethod, abstractstaticmethod

import numpy as np


class SysIdAlgBase(ABC):
    def __init__(self, data, y_name, u_name, settings):

        y_name = np.array(y_name) # now I can iterate over
        y=[]
        for name in y_name:
            y.append(data[name])
        self.y = np.array(y)

        u_name = np.array(u_name) # now I can iterate over
        u=[]
        for name in u_name:
            u.append(data[name])
        self.u = np.array(u)
        
        self.Ts = data.Ts
        self.settings = settings

    @abstractmethod
    def ident(self, order):
        pass

    @abstractstaticmethod
    def name():
        pass

    @staticmethod
    def _sse(y, y_hat):
        e = y - y_hat
        with np.errstate(over="ignore", invalid="ignore"):
            sse = e.T @ e
        return np.sum(sse)
