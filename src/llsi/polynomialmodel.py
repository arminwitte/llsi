#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:16:09 2022

@author: armin
"""
import numpy as np

from .ltimodel import LTIModel

class PolynomialModel(LTIModel):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.a = np.array()
        self.b = np.array()