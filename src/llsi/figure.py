"""
Plotting utilities for system identification.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as MplFigure
except ImportError:
    plt = None
    Axes = Any
    MplFigure = Any

from .ltimodel import LTIModel
from .statespacemodel import StateSpaceModel
from .sysidalgbase import compute_residuals_analysis
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
        if plt is None:
            raise ImportError("matplotlib is required for plotting. Install it with 'pip install llsi[plot]'.")

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
            "residuals": self._residuals_acf,
            "residuals_acf": self._residuals_acf,
            "residuals_ccf": self._residuals_ccf,
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

        # Only show the figure if matplotlib interactive mode is enabled.
        # In test environments (Agg backend) this avoids a non-interactive warning.
        try:
            if plt.isinteractive():
                plt.show()
        except Exception:
            # If anything unexpected occurs, avoid raising during normal program exit.
            pass

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

        if plot_type == "residuals":
            # Special case: residuals implies both ACF and CCF
            # We add two plots
            for o in obj_list:
                self.objects.append(o)
                self.plot_types.append("residuals_acf")
                self.place.append(self.counter)
                self.counter += 1

                self.objects.append(o)
                self.plot_types.append("residuals_ccf")
                self.place.append(self.counter)
                self.counter += 1
            return

        for o in obj_list:
            self.objects.append(o)
            self.plot_types.append(plot_type)
            self.place.append(self.counter)

        self.counter += 1

    @staticmethod
    def _impulse(fig: MplFigure, ax: Axes, lti_mod: LTIModel, col: str = "#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            res = lti_mod.impulse_response(N=200, uncertainty=True)
            if len(res) == 3:
                t, y, y_std = res
            else:
                t, y = res
                y_std = None

            markerline, stemlines, baseline = ax.stem(t, y)
            plt.setp(stemlines, "color", col)
            plt.setp(markerline, "color", col)

            if y_std is not None:
                y = y.ravel()
                y_std = y_std.ravel()
                # Plot confidence region around y=0 (significance band)
                ax.fill_between(t, -2 * y_std, 2 * y_std, color=col, alpha=0.2)

            ax.set_title("Impulse response")
            ax.grid(True, alpha=0.3)

    @staticmethod
    def _step(fig: MplFigure, ax: Axes, lti_mod: LTIModel, col: str = "#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            res = lti_mod.step_response(N=200, uncertainty=True)
            if len(res) == 3:
                t, y, y_std = res
            else:
                t, y = res
                y_std = None

            ax.step(t, y, where="post", color=col)

            if y_std is not None:
                y = y.ravel()
                y_std = y_std.ravel()
                try:
                    ax.fill_between(t, y - 2 * y_std, y + 2 * y_std, color=col, alpha=0.2, step="post")
                except (AttributeError, TypeError):
                    ax.fill_between(t, y - 2 * y_std, y + 2 * y_std, color=col, alpha=0.2)

            ax.set_title("Step response")
            ax.grid(True, alpha=0.3)

    @staticmethod
    def _frequency(fig: MplFigure, ax: Axes, lti_mod: LTIModel, col: str = "#1f77b4"):
        if isinstance(lti_mod, LTIModel):
            res = lti_mod.frequency_response(uncertainty=True)
            if len(res) == 4:
                omega, H, mag_std, phase_std = res
            else:
                omega, H = res
                mag_std = None
                phase_std = None

            H = H.squeeze()
            mag = np.abs(H.ravel())
            phase = np.angle(H.ravel())

            ax2 = ax.twinx()

            ax.plot(omega.ravel(), mag, color=col, label="Magnitude")

            if mag_std is not None:
                mag_std = mag_std.ravel()
                ax.fill_between(omega.ravel(), mag - 2 * mag_std, mag + 2 * mag_std, color=col, alpha=0.2)

            ax.set_ylabel("Magnitude")
            ax.set_xlabel("Frequency [rad/s]")
            ax.set_title("Frequency response")
            ax.grid(True, alpha=0.3)

            ax2.plot(omega, phase, linestyle="dashed", color=col, alpha=0.6, label="Phase")

            if phase_std is not None:
                phase_std = phase_std.ravel()
                ax2.fill_between(omega.ravel(), phase - 2 * phase_std, phase + 2 * phase_std, color=col, alpha=0.1)

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
        t = data.time

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

        t = data.time
        ax.plot(t, data[y_name], "--k", label="Measured", alpha=0.6)

        for i, m in enumerate(mods):
            # Cycle colors for models
            c = plt.rcParams["axes.prop_cycle"].by_key()["color"][(i) % 10]

            if isinstance(m, LTIModel):
                res = m.simulate(data[u_name], uncertainty=True)
                if len(res) == 2:
                    y_hat, y_std = res
                else:
                    y_hat = res
                    y_std = None
            else:
                y_hat = m.simulate(data[u_name])
                y_std = None

            fit = m.compare(data[y_name], data[u_name])
            ax.plot(t, y_hat, color=c, label=f"Model {i + 1} (Fit={fit * 100:.1f}%)")

            if y_std is not None:
                y_hat = y_hat.ravel()
                y_std = y_std.ravel()
                ax.fill_between(t, y_hat - 2 * y_std, y_hat + 2 * y_std, color=c, alpha=0.2)

        ax.set_title("Model Comparison")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_name)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _residuals_acf(fig: MplFigure, ax: Axes, obj: Dict[str, Any], col: str = "#1f77b4"):
        mod = obj.get("mod")
        data = obj.get("data")

        if mod is None or data is None:
            return

        # Handle list of models (take first one)
        if isinstance(mod, list):
            if len(mod) == 0:
                return
            mod = mod[0]

        try:
            res_analysis = compute_residuals_analysis(mod, data)
        except ValueError:
            return

        acf = res_analysis["acf"]
        lags = res_analysis["lags"]
        conf_interval = res_analysis["conf_interval"]

        # Handle NaNs
        if np.any(np.isnan(acf)):
            acf = np.nan_to_num(acf)

        mask = (lags > -50) & (lags < 50)

        ax.stem(lags[mask], acf[mask], markerfmt=".", label="ACF")
        ax.axhspan(-conf_interval, conf_interval, alpha=0.3, color="red")
        ax.set_title("Residuals ACF (Output 0)")
        ax.set_ylabel("ACF")
        ax.set_xlabel("Lag")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _residuals_ccf(fig: MplFigure, ax: Axes, obj: Dict[str, Any], col: str = "#1f77b4"):
        mod = obj.get("mod")
        data = obj.get("data")

        if mod is None or data is None:
            return

        # Handle list of models (take first one)
        if isinstance(mod, list):
            if len(mod) == 0:
                return
            mod = mod[0]

        try:
            res_analysis = compute_residuals_analysis(mod, data)
        except ValueError:
            return

        ccf = res_analysis["ccf"]
        lags = res_analysis["lags"]
        conf_interval = res_analysis["conf_interval"]

        # Handle NaNs
        if np.any(np.isnan(ccf)):
            ccf = np.nan_to_num(ccf)

        mask = (lags > -50) & (lags < 50)

        ax.stem(lags[mask], ccf[mask], markerfmt=".", linefmt="C1-", label="CCF")
        ax.axhspan(-conf_interval, conf_interval, alpha=0.3, color="red")
        ax.set_title("Residuals CCF (Output 0 vs Input 0)")
        ax.set_ylabel("CCF")
        ax.set_xlabel("Lag")
        ax.grid(True, alpha=0.3)
