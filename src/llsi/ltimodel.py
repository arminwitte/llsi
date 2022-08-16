#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 00:40:00 2021

@author: armin
"""

from abc import ABC, abstractmethod

import numpy as np


class LTIModel(ABC):
    def __init__(self, Ts=1.0):
        self.Ts = Ts
        self.info = {}

    def impulse_response(self, N=100, plot=False):
        t = np.linspace(0, (N - 1) * self.Ts, N)
        u = np.zeros((N,))
        u[0] = 1 / self.Ts

        y = self.simulate(u)

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(16, 9))
            ax.stem(t, y)
            plt.plot()

        return t, y

    def step_response(self, N=100, plot=False):
        t = np.linspace(0, (N - 1) * self.Ts, N)
        u = np.ones((N,))

        y = self.simulate(u)

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(16, 9))
            ax.stem(t, y)
            plt.plot()

        return t, y

    def compare(self, y, u):
        y_hat = self.simulate(u)
        return 1.0 - self.NRMSE(y, y_hat)

    @staticmethod
    def residuals(y, y_hat):
        return y.ravel() - y_hat.ravel()

    @staticmethod
    def SE(e):
        return np.power(e.ravel(), 2)

    @staticmethod
    def SSE(e):
        e_ = e.ravel()
        return e_.T @ e_

    @staticmethod
    def MSE(e):
        e_ = e.ravel()
        return 1 / len(e_) * e_.T @ e_
        return np.mean(LTIModel.SE(e_))

    @staticmethod
    def RMSE(e):
        return np.sqrt(LTIModel.MSE(e))

    @staticmethod
    def NRMSE(y, y_hat, normalization="matlab"):
        e = LTIModel.residuals(y, y_hat)
        if normalization in "matlab":
            nrmse = np.linalg.norm(e) / np.linalg.norm(y - np.mean(y))
        elif normalization in "mean":
            nrmse = LTIModel.RMSE(e) / np.mean(y)
        elif normalization in "ptp":
            nrmse = LTIModel.RMSE(e) / np.ptp(y)
        else:
            raise ValueError(f"Unknown normalization method {normalization}")
        return nrmse

    @abstractmethod
    def simulate(u):
        pass
