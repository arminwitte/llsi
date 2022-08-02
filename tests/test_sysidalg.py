#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:47 2021

@author: armin
"""

from llsi import sysid, sysidalg, SysIdData
from llsi.statespacemodel import StateSpaceModel
import scipy.stats
import numpy as np

def generate_data(filt = None, plot=False, noise=0.01):
    if filt is None:
        filt = StateSpaceModel.from_PT1(1,10.)
    
    if plot:
        filt.impulse_response(plot=True)

    t = np.linspace(0,9999*filt.Ts,10000)
    u = scipy.stats.norm.rvs(size=10000)
    e = noise * scipy.stats.norm.rvs(size=10000)
    
    y = filt.simulate(u) + e
    
    data = SysIdData(y=y,u=u,t=t)
    data.equidistant()
    
    if plot:
        data.show()
    return data

#-----------------------------------------------------------------------------

# data = generate_data(noise=0.1,plot=True)
# mod = sysid(data,'y','u',(1),method='n4sid')
# mod.impulse_response(plot=True)


#-----------------------------------------------------------------------------

# from subspace import N4SID

# n4sid = N4SID(data,'y','u')


#-----------------------------------------------------------------------------


filt = StateSpaceModel(A=[[0.8,0.8],[0,0.8]],B=[1,1],C=[1,0],D=0,Ts=1)
# filt = StateSpaceModel(A=[[0.8]],B=[1],C=[1],D=0,Ts=1)
ti, i = filt.impulse_response(plot=False)
data = generate_data(filt=filt,plot=False,noise=1.)
data.center()
mod1 = sysid(data,'y','u',(2),method='n4sid')
ti1, i1 = mod1.impulse_response(plot=False)
mod2 = sysid(data,'y','u',(2),method='po-moesp')
ti2, i2 = mod2.impulse_response(plot=False)

mod = mod1

import matplotlib.pyplot as plt
plt.close('all')
fig, ax = plt.subplots(2,2,figsize=(16,9))
data.plot(ax[0,0])
ax[1,0].stem(ti,i)
ax[0,1].stem(ti1,i1)
ax[1,1].stem(ti2,i2)

fig, ax = plt.subplots(figsize=(16,9))
mod.plot_hsv(ax)

plt.show()





