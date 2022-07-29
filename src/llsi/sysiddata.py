#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 19:40:17 2021

@author: armin
"""

import numpy as np
import scipy.interpolate
import scipy.signal

class SysIdData:
    def __init__(self,t=None,Ts=None,t_start=None,**kwargs):
        self.N = None
        self.series = {}
        self.add_series(**kwargs)
        self.t = t
        self.Ts = Ts
        
        if self.t is not None:
            self.t_start = t[0]
        else:
            self.t_start = t_start
            
    def __getitem__(self,key):
        return self.series[key]
            
    def add_series(self,**kwargs):
        for key, val in kwargs.items():
            s = np.array(val).ravel()
            self.series[key] = s
            
            if self.N is None:
                self.N = s.shape[0]
            else:
                if self.N != s.shape[0]:
                    # TODO: Throw error
                    print('ERROR')
                    
    def remove(self,key):
        del self.series[key]

    def time(self):
        if self.t is not None:
            return self.t
        else:
            t_start = self.t_start
            t_end = t_start + (self.N - 1) * self.Ts
            return np.linspace(t_start,t_end,self.N)
            
    def equidistant(self,N=None):
        if N is None:
            N = self.N
            
        if N < self.N:
            print('WARNING: Downsampling without filter!')
        
        t_ = self.time()
        
        t_start = t_[0]
        t_end = t_[-1]
            
        t = np.linspace(t_start,t_end,N)
        for key, val in self.series.items():
            f = scipy.interpolate.interp1d(self.t,val)
            self.series[key] = f(t)
        
        self.N = N
        self.Ts = (t_end - t_start)/(self.N - 1) 
        self.t = None
        
    def center(self):
        for key, val in self.series.items():
            self.series[key] -= np.mean(val)
        
    def plot(self,ax):
        t = self.time()
        
        for key, val in self.series.items():
            ax.plot(t,val,label=key)
        
    def show(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16,9))
        self.plot(ax)
        plt.show()
        
    def crop(self,start=0,end=-1):
        if self.t is not None:
            self.t = self.t[start:end]
        for key, val in self.series.items():
            self.series[key] = val[start:end]
            self.N = self.series[key].shape[0]
            
    def downsample(self,q):
        for key, val in self.series.items():
            self.series[key] = scipy.signal.decimate(val,q)
            self.N = self.series[key].shape[0]
            self.Ts *= q
        
        
    @classmethod
    def from_excel(cls,filename,column_names=None):
        import pandas as pd
        data = pd.read_excel(filename)
        d = {}
        for key in data.columns:
            if key in column_names or column_names is None:
                d[key] = data[key]
                N = data[key].values.shape[0]
        t = np.arange(0,N,1)
        sysiddata = cls(t=t,**d)
        return sysiddata
                    
            
        