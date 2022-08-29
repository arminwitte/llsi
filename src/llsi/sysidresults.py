#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:25:22 2022

@author: armin
"""

from typing import TypedDict

import numpy as np

from .polynomialmodel import PolynomialModel
from .statespacemodel import StateSpaceModel


class SysIdResultsDict(TypedDict):
    mod: StateSpaceModel | PolynomialModel
    residuals: np.ndarray
    message: str


# class SysIdResults:
#     def __init__(self, **kwargs):
#         self.mod = kwargs.get("mod")
#         self.residuals = kwargs.get("residuals")
#         self.message = kwargs.get("message")
