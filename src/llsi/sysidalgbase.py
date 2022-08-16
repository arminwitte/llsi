#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

from abc import ABC, abstractmethod, abstractstaticmethod

# import numpy as np


class SysIdAlgBase(ABC):
    def __init__(self, data, y_name, u_name, settings):
        self.y = data[y_name]
        self.u = data[u_name]
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
        return e.T @ e
