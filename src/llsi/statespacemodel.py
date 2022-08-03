#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 22:54:53 2021

@author: armin
"""

import numpy as np
import scipy.linalg

from .ltimodel import LTIModel

class StateSpaceModel(LTIModel):
    def __init__(self,A=None,B=None,C=None,D=None,Ts=1.,Nx=0):
        super().__init__(Ts = Ts)
        self.A = np.array(A)
        self.B = np.array(B).reshape(-1,1)
        self.C = np.array(C).reshape(1,-1)
        self.D = D
        
        if A is not None:
            self.Nx = self.A.shape[0]
        else:
            self.Nx = Nx
        
    def vectorize(self):
        theta = np.vstack([self.A.reshape(-1,1),
                 self.B.reshape(-1,1),
                 self.C.reshape(-1,1),
                 self.D])
        
        self.n = self.B.shape[0]
        
        return np.array(theta).ravel()
    
    def reshape(self,theta):
        n = self.n
        self.A = theta[:n*n].reshape(n,n)
        self.B = theta[n*n:n*n+n].reshape(n,1)
        self.C = theta[n*n+n:n*n+2*n].reshape(1,n)
        self.D = theta[-1]
    
    def simulate(self,u):
        u = u.ravel()
        # TODO: initialize x properly
        x1 = np.zeros((self.Nx,1))
        y = []
        for u_ in u:
            x = x1
            x1 = self.A @ x + self.B * u_
            y_ = self.C @ x + self.D * u_
            y.append(y_[0])
            
        return np.array(y).ravel()
    
    @classmethod
    def from_PT1(cls,K,tauC,Ts=1.):
        t = 2 * tauC
        tt = 1 / (Ts + t)
        b = K * Ts * tt
        a = (Ts - t) * tt
        
        B = [(1 - a) * b]
        D = b
        
        A = [[-a]]
        C = [1]
        
        mod = cls(A=A,B=B,C=C,D=D,Ts=Ts,Nx=1)
        
        return mod
    
    def plot_hsv(self,ax):
        hsv = self.info['Hankel singular values']
        ax.bar(np.arange(0,len(hsv),1),hsv)
        
    def to_ss(self,continuous=False,method='bilinear'):
        from scipy import signal
        if continuous:
            A, B, C, D = self._d2c(self.A, self.B, self.C, self.D, self.Ts, method=method)
            sys = signal.StateSpace(A,B,C,D)
        else:
            sys = signal.StateSpace(self.A, self.B, self.C, self.D, dt=self.Ts)
        return sys
    
    @staticmethod
    def _d2c(A,B,C,D,Ts,method='bilinear'):
        # https://math.stackexchange.com/questions/3820100/discrete-time-to-continuous-time-state-space
        if method in 'bilinear':
            return StateSpaceModel._d2c_bilinear(A,B,C,D,Ts)
        else:
            return StateSpaceModel._d2c_euler(A,B,C,D,Ts)
    
    @staticmethod
    def _d2c_bilinear(A,B,C,D,Ts):
        I = np.eye(*A.shape)
        AI = scipy.linalg.inv(A + I)
        A_ = 2.0/Ts * (A - I) @ AI
        B_ = 2.0/Ts * (I - (A - I) @ AI) @ B
        C_ = C @ AI
        D_ = D - C @ AI @ B
        return A_, B_, C_, D_
    
    @staticmethod
    def _d2c_euler(A,B,C,D,Ts):
        A_ = (A - np.eye(*A.shape))/Ts
        B_ = B/Ts
        C_ = C
        D_ = D
        return A_, B_, C_, D_
    
    def to_tf(self,continuous=False,method='bilinear'):
        sys = self.to_ss(continuous=continuous,method=method)
        return scipy.signal.TransferFunction(sys)
    
    def to_zpk(self,continuous=False,method='bilinear'):
        sys = self.to_ss(continuous=continuous,method=method)
        return scipy.signal.ZerosPolesGain(sys)
        
    def __repr__(self):
        s = f'A:\n{self.A}\n'
        s += f'B:\n{self.B}\n'
        s += f'C:\n{self.C}\n'
        s += f'D:\n{self.D}\n'
        return s
    
    def __str__(self):
        return self.__repr__()
    
    
        