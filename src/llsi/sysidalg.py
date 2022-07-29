#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

import numpy as np
from abc import ABC, abstractmethod, abstractstaticmethod
import scipy.optimize

class SysIdAlg(ABC):
    def __init__(self,data,y_name,u_name):
        self.y = data[y_name]
        self.u = data[u_name]
        self.Ts = data.Ts
    
    @abstractmethod
    def ident(self,order):
        pass
    
    @abstractstaticmethod
    def name():
        pass
    
    @staticmethod
    def _sse(y,y_hat):
        e = y - y_hat
        return np.sum(e**2)
    
class PEM_Poly(SysIdAlg):
    def __init__(self,data,y_name,u_name):
        super().__init__(data,y_name,u_name)
        
    def ident(self,order):
        pass
        
    @staticmethod
    def name():
        return 'pem_poly'
    
class PEM_SS(SysIdAlg):
    def __init__(self,data,y_name,u_name):
        super().__init__(data,y_name,u_name)
        alg = sysidalg.get_creator('po-moesp')
        # alg = sysidalg.get_creator('n4sid')
        self.alg_inst = alg(data,y_name,u_name)
        
    def ident(self,order):
        mod = self.alg_inst.ident(order)
        y_hat = mod.simulate(self.u)
        sse0 = self._sse(self.y,y_hat)
        
        def fun(x):
            mod.reshape(x)
            y_hat = mod.simulate(self.u)
            sse = self._sse(self.y,y_hat)
            print('{:10.6g}'.format(sse/sse0))
            return sse/sse0
    
        x0 = mod.vectorize()
        res = scipy.optimize.minimize(fun,x0)
        # res = scipy.optimize.minimize(fun,x0,method='nelder-mead')
        # res = scipy.optimize.minimize(fun,x0,method='BFGS')
        
        mod.reshape(res.x)
        
        return mod
        
    @staticmethod
    def name():
        return 'pem_ss'
    
class SysIdAlgFactory:
    def __init__(self):
        self.creators = {}
        
    def register_creator(self,creator):
        name = creator.name()
        self.creators[name] = creator
        
    def get_creator(self,name):
        c = self.creators[name]
        return c
    
sysidalg = SysIdAlgFactory()

from .subspace import N4SID, PO_MOESP
sysidalg.register_creator(N4SID)
sysidalg.register_creator(PO_MOESP)
sysidalg.register_creator(PEM_SS)

def sysid(data,y_name,u_name,order,method='bla'):
    alg = sysidalg.get_creator(method)
    alg_inst = alg(data,y_name,u_name)
    mod = alg_inst.ident(order)
    return mod
    