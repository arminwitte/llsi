#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:19:30 2021

@author: armin
"""
import numpy as np
from llsi.statespacemodel import StateSpaceModel

# ss = StateSpaceModel(A=[[0.9]],B=[1],C=[1],D=1,Ts=1)


# ss = StateSpaceModel.from_PT1(1, 10.0)

# t = np.linspace(0, 99, 100)
# u = np.zeros((100,))
# u[0] = 1.0

# y = ss.simulate(u)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(16, 9))

# ax.stem(t, y)

# plt.plot()
