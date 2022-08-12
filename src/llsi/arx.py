#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 01:34:47 2022

@author: armin
"""

from .sysidalg import SysIdAlg


class ARX(SysIdAlg):
    def __init__(self, data, y_name, u_name):
        pass

    def ident(self, order):
        pass

    @staticmethod
    def name():
        return "arx"
