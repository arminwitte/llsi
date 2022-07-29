#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:10:56 2021

@author: armin
"""

import numpy as np
import scipy.stats
import scipy.fft
import scipy.signal

class MCSigGen:
    def __init__(self,N=1000,A=1,f_cut=1000):
        self.N = N
        
    def generate(self):
        pass
    
    def _check(self):
        pass
    

def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = t * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return t, y

def fft(t,y):
    Ts = t[1] - t[0]
    Y = scipy.fft.rfft(y)
    f = scipy.fft.rfftfreq(t.shape[-1],Ts)
    return f, Y
    
duration = 1
fs = 1000
N = fs * duration
# t, y = generate_sine_wave(10,fs,duration)
# y = scipy.stats.norm.rvs(size=1000)

def mc(N=1000,f_cut=200,Ts = 0.001):
    y = [scipy.stats.norm.rvs()]
    t = np.linspace(0,(N-1)*Ts,N)
    j = 1
    y_ = 0.
    alpha = 0.2
    for i in range(100000):
        y_ = alpha * y_ + (1 - alpha) * scipy.stats.norm.rvs()
        y_test = y + [y_]
        # f, Y = fft(t[:j+1],y_test)
        f, Y = scipy.signal.welch(y_test,1/Ts)
        if any(Y[f>f_cut] > 0.02):
            pass
        else:
            j += 1
            y = y_test
            
        if j == N:
            break
            
    return t, y
    
t, y = mc()
# f, Y = fft(t, y)
f, Y = scipy.signal.welch(y,1/(t[1] - t[0]))

import matplotlib.pyplot as plt

plt.close('all')
fig, ax = plt.subplots(2,2,figsize=(16,9))
ax[0,0].plot(t,y)
ax[0,1].plot(f,np.abs(Y))
ax[1,1].plot(f,np.angle(Y))


plt.show()
    
    