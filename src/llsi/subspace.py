#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:54:55 2021

@author: armin
"""

from .sysidalg import SysIdAlg
from .statespacemodel import StateSpaceModel
import numpy as np
import scipy.linalg

class SubspaceIdent(SysIdAlg):
    def __init__(self,data,y_name,u_name):
        super().__init__(data,y_name,u_name)
        
    def hankel(self,x,n):
        A = []
        for i in range(n):
            A.append(x[i:-n+i].T)
            
        return np.array(A)
    
    @staticmethod
    def lq(A):
        Q, R = scipy.linalg.qr(A.T,mode='economic')
        return R.T, Q.T

class N4SID(SubspaceIdent):
    def __init__(self,data,y_name,u_name):
        super().__init__(data,y_name,u_name)
        
        # estimate extended observability matrix and states. Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)
        
    def ident(self,order):
        if isinstance(order,(tuple,list)):
            n = order[0]
        else:
            n = order
            
        r = 2*n+1 # window length
        
        Y = self.hankel(self.y,2*r)
        U = self.hankel(self.u,2*r)
        
        s = Y.shape[1]
        
        Yp = Y[0:r,:]
        Up = U[0:r,:]
        
        Yf = Y[r:2*r,:]
        Uf = U[r:2*r,:]
        
        Wp = np.vstack((Up,Yp))
        # Wf = np.vstack((Uf,Yf))
        Psi = np.vstack((np.vstack((Uf,Wp)),Yf))
        

        L, Q = self.lq(Psi)
        
        # L11 = L[  0:r  ,0:r]
        # L12 = L[  0:r  ,r:2*r]
        # L21 = L[  r:3*r,0:r]
        L22 = L[  r:3*r,r:3*r]
        # L31 = L[3*r:4*r,0:r]
        L32 = L[3*r:4*r,r:3*r]
        
        Gamma_r = L32 @ np.linalg.pinv(L22) @ Wp # oblique projection
        U_, s_, V_ = scipy.linalg.svd(Gamma_r,full_matrices=True)
        
        self.singular_values = s_

        # U1 = U_[:,0:n]
        # U2 = U_[:,n:r]
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))
        # Sigma_sqrt = scipy.linalg.diagsvd(s_, *Gamma_r.shape)
        V1 = V_[:,0:n]
        # V2 = V_[:,n:r]
        
        # Or = U1 @ Sigma_sqrt # extended observability matrix
        # print(Or)
        
        
        Xf = Sigma_sqrt @ V1.T # state matrix
        Y_ = np.vstack((Xf[:,1:s],self.y[r:r+s-1].T))
        X_ = np.vstack((Xf[:,0:s-1],self.u[r:r+s-1].T))
        Theta = Y_ @ np.linalg.pinv(X_)
        A = Theta[:n,:n]
        B = Theta[:n,n].ravel()
        C = Theta[n,:n].ravel()
        D = Theta[n,n]
        
        mod = StateSpaceModel(A=A,B=B,C=C,D=D,Ts=self.Ts)
        mod.info['Hankel singular values'] = s_
        
        return mod
        
        
    @staticmethod
    def name():
        return 'n4sid'

class PO_MOESP(SubspaceIdent):
    def __init__(self,data,y_name,u_name):
        super().__init__(data,y_name,u_name)
        
        # estimate extended observability matrix and states. Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)
        
    def ident(self,order):
        # Tangirala 2014
        # Algorithm 23.3
        if isinstance(order,(tuple,list)):
            n = order[0]
        else:
            n = order
        
        N = self.y.shape[0]
            
        r = 2*n+1 # window length
        
        Y = self.hankel(self.y,2*r)
        U = self.hankel(self.u,2*r)
        
        s = Y.shape[1]
        
        Yp = Y[0:r,:]
        Up = U[0:r,:]
        
        Yf = Y[r:2*r,:]
        Uf = U[r:2*r,:]
        
        Wp = np.vstack((Up,Yp))
        # Wf = np.vstack((Uf,Yf))
        Psi = 1./N * np.vstack((np.vstack((Uf,Wp)),Yf))
        # Psi = np.vstack((np.vstack((Uf,Wp)),Yf))

        L, Q = self.lq(Psi)
        
        L11 = L[  0:r  ,0:r]
        # L12 = L[  0:r  ,r:3*r]
        # L21 = L[  r:3*r,0:r]
        # L22 = L[  r:3*r,r:3*r]
        L31 = L[3*r:4*r,0:r]
        L32 = L[3*r:4*r,r:3*r]
        
        # Gamma_r = 1./np.sqrt(N) * L32
        Gamma_r = L32
        U_, s_, V_ = scipy.linalg.svd(Gamma_r,full_matrices=False)
        
        self.singular_values = s_

        # print(s_)

        U1 = U_[:,0:n]
        U2 = U_[:,n:r]
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))
        # V1 = V_[:,0:n]
        # V2 = V_[:,n:r]
        
        Or = U1# @ Sigma_sqrt # extended observability matrix
        
        C = Or[0,:] # TODO: might be wrong!!!
        A = scipy.linalg.pinv(Or[0:-1,:]) @ Or[1:,:]
        # A = scipy.linalg.lstsq(Or[0:-1,:],Or[1:,:])
        # print(A)
        
        P = U2.T
        # print(P)
        A1_ = P.ravel(order='F')
        
        nn = A1_.shape[0]
        ny = 1
        A_ = np.zeros((nn,ny+n))
        A_[:,0:ny] = A1_.reshape(-1,ny)
                
        for i in range(1,r):
            Pi = P[:,i:r]
            Oi = Or[0:r-i,:]
            Ni = Pi @ Oi
            j = (i - 1) * (r - n)
            A_[j:j+Ni.shape[0],ny:ny+Ni.shape[1]] = Ni
            
        # print(A_)
            
        M = (U2.T @ L31 @ np.linalg.inv(L11)).ravel(order='F')
        
        x_, *_ = scipy.linalg.lstsq(A_,M)
        # x_ = scipy.linalg.pinv(A_) @ M
        
        D = x_[0:ny]
        B = x_[ny:ny+n]

        
        mod = StateSpaceModel(A=A,B=B,C=C,D=D,Ts=self.Ts)
        mod.info['Hankel singular values'] = s_
        
        return mod
        
        
    @staticmethod
    def name():
        return 'po-moesp'