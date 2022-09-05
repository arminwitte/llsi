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
data.center()
data.downsample(30)
data ,test_set = data.split(0.6)
data.lowpass(1, 10)
data.crop(start=100)
data.center()
test_set.center()

with llsi.Figure() as fig:
    fig.plot(data)
    res1 = llsi.sysid(data,'Nu','Re',(2,),method='po-moesp')
    res2 = llsi.sysid(data,'Nu','Re',(2,),method='n4sid')
    mod1 = res1.mod
    mod2 = res2.mod
    fig.plot([mod1,mod2],'impulse')
    fig.plot([mod1,mod2],'step')
    fig.plot([mod1,mod2],'hsv')
    fig.plot({"mod":[mod1,mod2],"data":test_set,"y_name":"Nu","u_name":"Re"},'compare')
