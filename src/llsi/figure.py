"""
Plotting utilities for system identification.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

from .ltimodel import LTIModel
from .statespacemodel import StateSpaceModel
from .sysiddata import SysIdData


class Figure:
    """
    Context manager for creating subplots of system identification results.
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 9)):
        """
        Initialize the Figure context manager.

        Args:
            figsize: Tuple of (width, height) for the figure.
        """
        self.objects: List[Any] = []
        self.plot_types: List[Optional[str]] = []
        self.place: List[int] = []

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
        self.fig: Optional[MplFigure] = None
        self.ax: Optional[Union[Axes, np.ndarray]] = None

    def __enter__(self) -> "Figure":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            return False  # Propagate exception

        rows = int(np.floor((self.counter + 1) / 2))
        cols = 1 if self.counter < 2 else 2

        # Handle case where no plots were added
        if rows == 0:
            return

        self.fig, self.ax = plt.subplots(rows, cols, figsize=self.figsize, constrained_layout=True)

        # Ensure self.ax is always indexable for consistency if possible,
        # but matplotlib returns Axes or array of Axes.

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
                        f"for an object of type {type(obj)} a plot_type has to be specified explicitly!"
                    )
                    continue
            else:
                fun = self.registry.get(plot_type)
                if fun is None:
                    self.logger.warning(f"Unknown plot_type '{plot_type}'. Skipping.")
                    continue

            # handle indexing of axes with different array sizes
            if self.counter == 1:
                ax = self.ax
            elif self.counter == 2:
                ax = self.ax[ind]
            else:
                # For 2 columns, row = ind // 2, col = ind % 2
                ax = self.ax[ind // 2, ind % 2]

            # Determine color based on object identity or index
            # This logic is a bit fragile but kept from original
            try:
                # Find index of this object in the list of unique objects added so far?
                # Or just use the index in the current list?
                # Original code: color_index = self.objects.index(obj)
                # This finds the *first* occurrence.
                color_index = self.objects.index(obj) % len(self.colors)
            except ValueError:
                color_index = 0

            # call plotting method
            fun(self.fig, ax, obj, col=self.colors[color_index])

        plt.show()

    def plot(self, obj: Union[Any, List[Any]], plot_type: Optional[str] = None):
        """
        Add an object to be plotted.

        Args:
            obj: The object(s) to plot (LTIModel, SysIdData, etc.)
            plot_type: The type of plot ('impulse', 'step', 'frequency', 'hsv', 'time_series', 'compare').
                       If None, inferred from object type.
        """
        if not isinstance(obj, (list, tuple)):
            obj_list = [obj]
        else:
            obj_list = obj

        for o in obj_list:
            self.objects.append(o)
            self.plot_types.append(plot_type)
            self.place.append(self.counter)

        self.counter += 1

    @staticmethod
    def _impulse(fig: MplFigure, ax: Axes, lti_mod: LTIModel, col: str = "#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            t, y = lti_mod.impulse_response(N=200)
            markerline, stemlines, baseline = ax.stem(t, y)
            plt.setp(stemlines, "color", col)
            plt.setp(markerline, "color", col)
            ax.set_title("Impulse response")
            ax.grid(True, alpha=0.3)

    @staticmethod
    def _step(fig: MplFigure, ax: Axes, lti_mod: LTIModel, col: str = "#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            t, y = lti_mod.step_response(N=200)
            ax.step(t, y, where="post", color=col)
            ax.set_title("Step response")
            ax.grid(True, alpha=0.3)

    @staticmethod
    def _frequency(fig: MplFigure, ax: Axes, lti_mod: LTIModel, col: str = "#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            omega, H = lti_mod.frequency_response()
            H = H.squeeze()

            ax2 = ax.twinx()

            ax.plot(omega.ravel(), np.abs(H.ravel()), color=col, label="Magnitude")
            ax.set_ylabel("Magnitude")
            ax.set_xlabel("Frequency [rad/s]")
            ax.set_title("Frequency response")
            ax.grid(True, alpha=0.3)

            ax2.plot(omega, np.angle(H.ravel()), linestyle="dashed", color=col, alpha=0.6, label="Phase")
            ax2.set_ylabel("Phase [rad]")
            ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax2.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    @staticmethod
    def _hsv(fig: MplFigure, ax: Axes, ss_mod: StateSpaceModel, col: str = "#1f77b4"):
        if isinstance(ss_mod, StateSpaceModel):
            hsv = ss_mod.info.get("Hankel singular values")
            if hsv is not None:
                hsv_scaled = hsv / np.sum(hsv)
                ax.bar(np.arange(1, len(hsv_scaled) + 1), hsv_scaled, color=col)
                ax.set_title("Hankel Singular Values")
                ax.set_xlabel("State")
                ax.set_ylabel("Normalized HSV")
            else:
                ax.text(0.5, 0.5, "No HSV data available", ha="center", va="center")

    @staticmethod
    def _time_series(fig: MplFigure, ax: Axes, data: SysIdData, col: str = "#1f77b4"):
        t = data.time()

        for key, val in data.series.items():
            ax.plot(t, val, label=key)  # Use default colors for multiple series

        ax.set_title("Time series")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _compare(fig: MplFigure, ax: Axes, obj: Dict[str, Any], col: str = "#1f77b4"):
        mods = obj.get("mod", [])
        if not isinstance(mods, list):
            mods = [mods]

        data = obj.get("data")
        y_name = obj.get("y_name")
        u_name = obj.get("u_name")

        if data is None or y_name is None or u_name is None:
            return

        t = data.time()
        ax.plot(t, data[y_name], "--k", label="Measured", alpha=0.6)

        for i, m in enumerate(mods):
            # Cycle colors for models
            c = plt.rcParams["axes.prop_cycle"].by_key()["color"][(i) % 10]

            y_hat = m.simulate(data[u_name])
            fit = m.compare(data[y_name], data[u_name])
            ax.plot(t, y_hat, color=c, label=f"Model {i + 1} (Fit={fit:.1f}%)")

        ax.set_title("Model Comparison")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_name)
        ax.grid(True, alpha=0.3)
