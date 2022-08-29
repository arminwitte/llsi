#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:38:26 2022

@author: armin
"""

import matplotlib.pyplot as plt
import numpy as np

from .ltimodel import LTIModel
from .statespacemodel import StateSpaceModel
from .sysiddata import SysIdData


class Figure:
    def __init__(self, figsize=(16, 9)):
        self.objects = []
        self.plot_types = []
        self.place = []

        self.registry = {
            "impulse": self._impulse,
            "step": self._step,
            "hsv": self._hsv,
            "time_series": self._time_series,
            "compare": self._compare,
            "autocorr": self._autocorr,
            "crosscorr": self._crosscorr,
        }

        self.figsize = figsize
        self.counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.fig, self.ax = plt.subplots(
            int(np.floor((self.counter + 1) / 2)),
            1 if self.counter < 2 else 2,
            figsize=self.figsize,
        )
        for i in range(len(self.objects)):
            plot_type = self.plot_types[i]
            obj = self.objects[i]
            ind = self.place[i]

            # handle default cases
            if not plot_type:
                # deduce type
                if isinstance(obj, SysIdData):
                    fun = self.registry["time_series"]
                elif isinstance(obj, LTIModel):
                    fun = self.registry["impulse"]
                else:
                    print(
                        f"for an object of type {type(obj)} a plot_type has to be specified explicitly!"
                    )
            else:
                fun = self.registry[plot_type]

            # handle indexing of axes with different array sizes
            if self.counter == 1:
                ax = self.ax
            elif self.counter == 2:
                ax = self.ax[ind]
            else:
                ax = self.ax[int(np.floor(ind / 2)), ind % 2]

            # call plotting method
            fun(self.fig, ax, obj)

        plt.plot()

    def plot(self, obj, plot_type=None):
        if not isinstance(obj, (list, tuple)):
            obj = [obj]

        for o in obj:
            self.objects.append(o)
            self.plot_types.append(plot_type)
            self.place.append(self.counter)

        self.counter += 1

    @staticmethod
    def _impulse(fig, ax, lti_mod):
        if isinstance(lti_mod, LTIModel):
            t, y = lti_mod.impulse_response()
            ax.stem(t, y)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            # ax.set_ylabel('Scores')
            ax.set_title("Impulse response")
            # ax.set_xticks(x, labels)
            # ax.legend()

    @staticmethod
    def _step(fig, ax, lti_mod):
        if isinstance(lti_mod, LTIModel):
            t, y = lti_mod.step_response()
            ax.step(t, y)
            ax.set_title("Step response")

    @staticmethod
    def _hsv(fig, ax, ss_mod):
        if isinstance(ss_mod, StateSpaceModel):
            hsv = ss_mod.info["Hankel singular values"]
            hsv_scaled = hsv / np.sum(hsv)
            ax.bar(np.arange(0, len(hsv_scaled), 1), hsv_scaled)
            ax.set_title("Hankel Singular Values")

    @staticmethod
    def _time_series(fig, ax, data):
        t = data.time()

        for key, val in data.series.items():
            ax.plot(t, val, label=key)

        ax.set_title("Time series")
        ax.legend()
        ax.set_ylabel("time")

    @staticmethod
    def _compare(fig, ax, obj):

        mods = obj.get("mod")
        data = obj.get("data")
        y_name = obj.get("y_name")
        u_name = obj.get("u_name")

        t = data.time()
        ax.plot(t, data[y_name], label="orig.")

        for m in mods:
            y_hat = m.simulate(data[u_name])
            fit = m.compare(data[y_name], data[u_name])
            ax.plot(t, y_hat, label=f"model (NRMSE-fit={fit:.3f})")

        ax.set_title("Comparison")
        ax.legend()
        ax.set_ylabel("time")
        ax.set_ylabel(y_name)

    @staticmethod
    def _autocorr(fig, ax, obj):

        mods = obj.get("mod")
        data = obj.get("data")
        y_name = obj.get("y_name")
        u_name = obj.get("u_name")

        for m in mods:
            y_hat = m.simulate(data[u_name])
            e = m.residuals(data[y_name], y_hat)
            lags, corr = m.autocorrelation(e)
            ax.stem(lags, corr, label="autocorrelation")

        ax.set_title("Comparison")
        ax.legend()
        ax.set_ylabel("time")
        ax.set_ylabel(y_name)

    @staticmethod
    def _crosscorr(fig, ax, obj):

        mods = obj.get("mod")
        data = obj.get("data")
        y_name = obj.get("y_name")
        u_name = obj.get("u_name")

        for m in mods:
            y_hat = m.simulate(data[u_name])
            e = m.residuals(data[y_name], y_hat)
            lags, corr = m.crosscorrelation(e, y_hat)
            ax.stem(lags, corr, label="crosscorrelation")

        ax.set_title("Crosscorrelation")
        ax.legend()
        # ax.set_ylabel("time")
        ax.set_xlabel("lag")
        # ax.set_ylabel(y_name)
