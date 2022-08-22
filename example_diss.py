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
data.equidistant(305002)
print(data.time().shape)
data.center()
data.downsample(18)
data.lowpass(1, 10)
data, test_set = data.split(0.8)
data.crop(start=1000)
data.center()
print(1/data.Ts)

import matplotlib.pyplot as plt


with llsi.Figure() as fig:
    mod1 = llsi.sysid(data,'Nu','Re',(0,100,0),method='arx',settings={"lambda":1e2})
    mod2 = llsi.sysid(data,'Nu','Re',(3,3,0),method='pem',settings={"init":"arx"})
    fig.plot([mod1,mod2],'impulse')
    fig.plot({"mod":[mod1,mod2],"data":test_set,"y_name":"Nu","u_name":"Re"},'compare')

if False:
    import scipy
    t, y = scipy.signal.dimpulse(mod1.to_ss())
    
    fig, ax = plt.subplots()
    plt.plot(t, np.squeeze(y))
    plt.grid()
    plt.xlabel('n [samples]')
    plt.ylabel('Amplitude')
    
    
    t, y = scipy.signal.impulse(mod1.to_ss(continuous = True))
    
    # fig, ax = plt.subplots()
    plt.plot(t[1:], np.squeeze(y)[1:])
    plt.grid()
    plt.xlabel('n [samples]')
    plt.ylabel('Amplitude')
    
    
    t, y = scipy.signal.impulse(mod1.to_ss(continuous = True,method='euler'))
    
    # fig, ax = plt.subplots()
    plt.plot(t[1:], np.squeeze(y)[1:])
    plt.grid()
    plt.xlabel('n [samples]')
    plt.ylabel('Amplitude')
    
    t, y = scipy.signal.dimpulse(mod1.to_tf())
    plt.plot(t[1:], np.squeeze(y)[1:])
    plt.grid()
    plt.xlabel('n [samples]')
    plt.ylabel('Amplitude')
    
    t, y = scipy.signal.dimpulse(mod1.to_zpk())
    plt.plot(t[1:], np.squeeze(y)[1:])
    plt.grid()
    plt.xlabel('n [samples]')
    plt.ylabel('Amplitude')
    
    
    
    
    
    plt.legend(['discrete','cont','euler','tf', 'zpk'])
    

fig, ax = plt.subplots()
ti1, i1 = mod1.impulse_response()
ti2, i2 = mod2.impulse_response(300)
plt.plot(ti1[50:], i1[50:])
plt.plot(ti2[50:], i2[50:],'r')
plt.grid(True)

print(i1[:5])
# print(i2[:5])