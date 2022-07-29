#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:25:46 2021

@author: armin
"""

from sysiddata import SysIdData

data = SysIdData(y=[1,2,3,4,5,6],u=[1,4,9,16,25,36],t=[1,1.5,2.5,3.0,3.3,4.1])


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16,9))
data.plot(ax)
data.equidistant()
data.plot(ax)
ax.legend()