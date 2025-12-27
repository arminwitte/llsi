#!/usr/bin/env python3
"""
Created on Wed Aug 17 09:38:26 2022

@author: armin
"""

import logging

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
            "frequency": self._frequency,
            "hsv": self._hsv,
            "time_series": self._time_series,
            "compare": self._compare,
        }

        self.figsize = figsize
        self.counter = 0
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.logger = logging.getLogger(__name__)

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
                    self.logger.warning(
                        f"for an object of type {type(obj)} a plot_type has to " + "be specified explicitly!"
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

            # get the index of the firts occurence of an object
            # SERIOUS FLAWS GUARANTEED!!!
            try:
                color_index = self.objects.index(obj)
            except ValueError:
                color_index = 0

            # call plotting method
            fun(self.fig, ax, obj, col=self.colors[color_index + 1])

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
    def _impulse(fig, ax, lti_mod, col="#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            t, y = lti_mod.impulse_response(N=200)
            markerline, stemlines, baseline = ax.stem(t, y)
            plt.setp(stemlines, "color", col)
            plt.setp(markerline, "color", col)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            # ax.set_ylabel('Scores')
            ax.set_title("Impulse response")
            # ax.set_xticks(x, labels)
            # ax.legend()

    @staticmethod
    def _step(fig, ax, lti_mod, col="#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            t, y = lti_mod.step_response(N=200)
            ax.step(t, y, color=col)
            ax.set_title("Step response")

    @staticmethod
    def _frequency(fig, ax, lti_mod, col="#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            omega, H = lti_mod.frequency_response()
            H = H.squeeze()  # TODO: not very nice

            ax2 = ax.twinx()

            # fig, ax = plt.subplots(2)
            ax.plot(omega.ravel(), np.abs(H.ravel()), color=col)
            ax.set_ylim(0, None)
            ax.set_ylabel("Magnitude")
            ax.set_xlabel("Frequency")
            # ax.grid()
            ax.set_title("Frequency response")

            ax2.plot(omega, np.angle(H.ravel()), linestyle="dashed", color=col)
            ax2.set_ylim(-np.pi, np.pi)
            ax2.set_ylabel("Phase in rad")
            ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            # ax2.grid()

            # fig.tight_layout()

    @staticmethod
    def _hsv(fig, ax, ss_mod, col="#1f77b4"):
        if isinstance(ss_mod, StateSpaceModel):
            hsv = ss_mod.info["Hankel singular values"]
            hsv_scaled = hsv / np.sum(hsv)
            ax.bar(np.arange(0, len(hsv_scaled), 1), hsv_scaled, color=col)
            ax.set_title("Hankel Singular Values")

    @staticmethod
    def _time_series(fig, ax, data, col="#1f77b4"):
        t = data.time()

        for key, val in data.series.items():
            ax.plot(t, val, label=key, color=col)

        ax.set_title("Time series")
        ax.legend()
        ax.set_ylabel("time")

    @staticmethod
    def _compare(fig, ax, obj, col="#1f77b4"):
        mods = obj.get("mod")
        data = obj.get("data")
        y_name = obj.get("y_name")
        u_name = obj.get("u_name")

        t = data.time()
        ax.plot(t, data[y_name], "--k", label="orig.")

        for m in mods:
            y_hat = m.simulate(data[u_name])
            fit = m.compare(data[y_name], data[u_name])
            ax.plot(t, y_hat, label=f"model (NRMSE-fit={fit:.4f})")

        ax.set_title("Comparison")
        ax.legend()
        ax.set_ylabel("time")
        ax.set_ylabel(y_name)
