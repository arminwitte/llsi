#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 00:40:00 2021

@author: armin
"""

from abc import ABC
import numpy as np

class LTIModel(ABC):
    def __init__(self,Ts=1.):
        self.Ts = Ts
        self.info = {}
        
    def impulse_response(self,N=100,plot=False):
        t = np.linspace(0,(N-1)*self.Ts,N)
        u = np.zeros((N,))
        u[0] = 1/self.Ts
        
        y = self.simulate(u)
        
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(16,9))
            ax.stem(t,y)
            plt.plot()
        
        return t, y
        
    def step_response(self,N=100,plot=False):
        t = np.linspace(0,(N-1)*self.Ts,N)
        u = np.ones((N,))
        
        y = self.simulate(u)
        
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(16,9))
            ax.stem(t,y)
            plt.plot()
        
        return t, y
        