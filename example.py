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
data.downsample(3)
data.crop(start=100)
data.center()

import matplotlib.pyplot as plt


# mod1 = llsi.sysid(data,'Nu','Re',(3,),method='n4sid')
# ti1, i1 = mod1.impulse_response(plot=True)
# fig, ax = plt.subplots()
# mod1.plot_hsv(ax)

mod1 = llsi.sysid(data,'Nu','Re',(3,),method='po-moesp')
ti1, i1 = mod1.impulse_response(plot=True)
fig, ax = plt.subplots()
mod1.plot_hsv(ax)

# mod1 = llsi.sysid(data,'Nu','Re',(3,),method='pem_ss')
# ti1, i1 = mod1.impulse_response(plot=True)
# fig, ax = plt.subplots()
# mod1.plot_hsv(ax)

# print(mod1.to_controllable_form())

# ss = mod1.to_ss()
# print(ss)

# tf = ss.to_tf()
# zpk = ss.to_zpk()
# ss_ = zpk.to_ss()
# print(ss_)

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
    
    
    
nrmse_fit = mod1.compare(data['Nu'],data['Re'])
print(f"NRMSE-fit: {nrmse_fit}")




# mod2, hsv = mod1.reduce_order(2)


fig, ax = plt.subplots()
ti1, i1 = mod1.impulse_response(plot=False)
# ti2, i2 = mod2.impulse_response(plot=False)
plt.plot(ti1, i1)
# plt.plot(ti2, i2,'r')

print(i1[:5])
# print(i2[:5])