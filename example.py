#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 23:10:34 2022

@author: armin
"""

import numpy as np

d = np.load(r'data/heated_wire_data.npy')

import llsi

t = d[:,0]
Re = d[:,1]
Nu = d[:,2]

data = llsi.SysIdData(t=t,Re=Re,Nu=Nu)
data.equidistant()
data ,test_set = data.split(0.6)
data.center()
data.downsample(30)
data.lowpass(1, 10)
data.crop(start=100)
data.center()
test_set.center()

with llsi.Figure() as fig:
    fig.plot([data])
    mod1 = llsi.sysid(data,'Nu','Re',(2,),method='po-moesp')
    mod2 = llsi.sysid(data,'Nu','Re',(2,),method='n4sid')
    fig.plot([mod1,mod2],'impulse')
    fig.plot([mod1,mod2],'step')
    fig.plot([mod1,mod2],'hsv')
    
nrmse_fit = mod1.compare(data['Nu'],data['Re'])
print(f"NRMSE-fit: {nrmse_fit}")
nrmse_fit = mod2.compare(data['Nu'],data['Re'])
print(f"NRMSE-fit: {nrmse_fit}")
    
nrmse_fit = mod1.compare(test_set['Nu'],test_set['Re'])
print(f"NRMSE-fit: {nrmse_fit}")
nrmse_fit = mod2.compare(test_set['Nu'],test_set['Re'])
print(f"NRMSE-fit: {nrmse_fit}")
