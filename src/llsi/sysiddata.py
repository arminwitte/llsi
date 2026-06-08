"""
Data container for system identification.
"""

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import scipy.interpolate
import scipy.signal

# PRBS helpers live in math.py (numba-decorated functions centralized there)
from . import math as _math


@dataclass
class SysIdData:
    """Container for time-series data used in system identification.

    Features:
    - Uses dataclasses for concise initialization
    - Supports equidistant and non-equidistant time sampling
    - Method chaining for fluent API design
    - Flexible data manipulation (crop, resample, filter, differentiate)
    - Pythonic interface (slicing, iteration, len)

    Examples:
        >>> # Create equidistant data
        >>> data = SysIdData(Ts=0.01, u=u_array, y=y_array)
        >>> # Slicing works like a list/array
        >>> train_data = data[:1000]
        >>> # Method chaining
        >>> data.equidistant(N=500).lowpass(order=4, corner_frequency=10).plot()
    """

    series: dict[str, np.ndarray] = field(default_factory=dict)
    t: Optional[np.ndarray] = None
    Ts: Optional[float] = None
    t_start: float = 0.0
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    trend_slopes: dict[str, float] = field(default_factory=dict)
    trend_intercepts: dict[str, float] = field(default_factory=dict)

    def __init__(
        self, t: Optional[np.ndarray] = None, Ts: Optional[float] = None, t_start: Optional[float] = None, **kwargs: Any
    ):
        # Compatibility constructor: accept either `series` dict or individual series kwargs
        series_arg = kwargs.pop("series", None)
        if series_arg is not None:
            if not isinstance(series_arg, dict):
                raise TypeError("series must be a dict of name: array")
            self.series = {k: np.asarray(v).ravel() for k, v in series_arg.items()}
        else:
            # remaining kwargs are interpreted as series
            self.series = {k: np.asarray(v).ravel() for k, v in kwargs.items()}

        # coerce time to numpy array if provided
        self.t = np.asarray(t) if t is not None else None
        self.Ts = Ts
        if t_start is not None:
            self.t_start = t_start
        else:
            if self.t is not None and self.t.size > 0:
                self.t_start = float(self.t[0])

        # Initialize scaling state fields
        self.means = {}
        self.stds = {}
        self.trend_slopes = {}
        self.trend_intercepts = {}

        # Call post-init validations and conversions
        self.__post_init__()

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Validate Ts
        if self.Ts is not None and self.Ts <= 0:
            raise ValueError(f"Sampling time Ts must be positive, got {self.Ts}")

        # Ensure series arrays are numpy arrays and set N
        for k, v in self:
            self.series[k] = np.asarray(v).ravel()
        if self.Ts is None and self.t is None:
            raise ValueError("Either 't' (time vector) or 'Ts' (sampling time) must be provided.")

    @property
    def N(self) -> int:
        """Number of samples inferred from stored series. Returns 0 if no series."""
        if not self.series:
            return 0
        return next(iter(self.series.values())).shape[0]

    @property
    def time(self) -> np.ndarray:
        """Time vector property. If `t` is None, construct from `t_start` and `Ts`."""
        if self.t is not None:
            return np.asarray(self.t)
        if self.N == 0:
            return np.array([])
        return self.t_start + np.arange(self.N) * self.Ts

    def __getitem__(self, key: Union[str, slice, int]) -> Union[np.ndarray, "SysIdData"]:
        """
        Get a series by name OR slice the dataset by index.

        Examples:
            data["u"]      -> Returns numpy array of series 'u'
            data[0:100]    -> Returns a NEW SysIdData object cropped to first 100 samples
            data[:50]      -> First 50 samples
        """
        if isinstance(key, str):
            return self.series[key]
        elif isinstance(key, (slice, int)):
            # Handle slicing logic. If int, we treat it as a single-point slice or fail?
            # Usually for time series objects, int access isn't well defined (row vs col).
            # We treat int as a slice of length 1 for consistency, or let crop handle it.
            start = key.start if isinstance(key, slice) else key
            stop = key.stop if isinstance(key, slice) else key + 1
            step = key.step if isinstance(key, slice) else 1

            if step is not None and step != 1:
                # Use downsample logic if step > 1? For now, standard slicing.
                # Standard array slicing supports steps, so we can support it manually
                # but crop() is range based. Let's use crop for contiguous slices.
                pass

            # If it's a simple contiguous slice, use crop (safer for metadata)
            if step is None or step == 1:
                return self.crop(start=start, end=stop, inplace=False)
            else:
                # Complex slicing (e.g. ::2)
                # We can implement this via direct array slicing
                target = copy.deepcopy(self)
                if target.t is not None:
                    target.t = target.t[key]
                    if len(target.t) > 0:
                        target.t_start = float(target.t[0])
                else:
                    # Update Ts and start if stepping
                    if start:
                        target.t_start += target.Ts * start
                    target.Ts *= step

                for k, v in target:
                    target.series[k] = v[key]
                return target

        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __repr__(self) -> str:
        """String representation for Jupyter notebooks and REPL."""
        series_names = list(self.series.keys())
        time_info = ""
        if self.N > 0:
            time_end = self.time[-1]
            time_info = f"  Time: {self.t_start:.2f} to {time_end:.2f}s"
        else:
            time_info = "  Time: empty"

        if self.Ts is not None:
            header = f"SysIdData(N={self.N}, Ts={self.Ts:.4f}s)"
        else:
            header = f"SysIdData(N={self.N}, Non-equidistant)"

        series_info = f"  Series: {', '.join(series_names)}" if series_names else "  Series: (none)"

        return "\n".join([header, time_info, series_info])

    def copy(self) -> "SysIdData":
        """Return a deep copy of this SysIdData object."""
        return copy.deepcopy(self)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.N

    def __contains__(self, key: str) -> bool:
        """Check if a series exists in the dataset."""
        return key in self.series

    def __iter__(self):
        """Iterate over (key, array) tuples of the series."""
        return iter(self.series.items())

    def add_series(self, **kwargs: Any) -> "SysIdData":
        """
        Add time series to the dataset.

        Args:
            **kwargs: Time series data as keyword arguments (name=data).
        """
        for key, val in kwargs.items():
            s = np.asarray(val).ravel()
            if self.series and s.shape[0] != self.N:
                raise ValueError(
                    f"Length of vector to add ({s.shape[0]}) does not match existing series length ({self.N})"
                )

            # Warn on duplicate keys
            if key in self.series:
                self.logger.warning(f"Series '{key}' already exists. Overwriting.")

            # Check for NaN (Warning only, do not block)
            if len(s) > 0 and np.isnan(s).any():
                self.logger.warning(f"Series '{key}' contains NaN values.")

            self.series[key] = np.asarray(s)
        return self

    def remove(self, key: str) -> "SysIdData":
        """Remove a time series and return self for chaining."""
        del self.series[key]
        return self

    # `time` property implemented above

    def equidistant(
        self, N: Optional[int] = None, inplace: bool = True, method: Union[str, dict[str, str]] = "linear"
    ) -> "SysIdData":
        """
        Resample data to be equidistant.

        Modifies the object in-place by default.

        Args:
            N: Number of points for the new grid. If None, keeps current N.
            inplace: If True, modify in-place. If False, return a copy.
            method: Interpolation method. Can be:
                - str: Single method applied to all series (e.g., "linear", "previous", "cubic").
                  "previous" is equivalent to Zero-Order Hold (ZOH), suitable for step inputs.
                - dict: Mapping series name to method, e.g., {"u": "previous", "y": "linear"}.
                  Uses default "linear" for unmapped series.

        Returns:
            SysIdData: The resampled object (self if inplace=True, copy if inplace=False).
        """
        target = self if inplace else copy.deepcopy(self)

        # Validate interpolation methods
        valid_methods = {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next"}
        if isinstance(method, str):
            if method not in valid_methods:
                raise ValueError(f"Invalid interpolation method '{method}'. Must be one of {valid_methods}")
            method_dict = dict.fromkeys(target.series.keys(), method)
        else:
            method_dict = {}
            for k in target.series.keys():
                m = method.get(k, "linear")
                if m not in valid_methods:
                    raise ValueError(f"Invalid interpolation method '{m}' for series '{k}'")
                method_dict[k] = m

        if N is None:
            N = target.N

        if N < target.N:
            target.logger.warning("Downsampling without filter! Aliasing may occur.")

        t_current = np.asarray(target.time)
        if t_current.size == 0:
            return target

        # Validate that time vector is strictly increasing
        if not np.all(np.diff(t_current) > 0):
            raise ValueError("Time vector must be strictly increasing for interpolation")

        t_start = t_current[0]
        t_end = t_current[-1]
        t_new = np.linspace(t_start, t_end, N)

        if target.series:
            keys = list(target.series.keys())

            # Resample each series with its specified method
            for k in keys:
                f = scipy.interpolate.interp1d(
                    t_current, target.series[k], kind=method_dict[k], fill_value="extrapolate"
                )
                target.series[k] = f(t_new)

        target.Ts = (t_end - t_start) / (N - 1) if N > 1 else None
        target.t = None
        return target

    def center(self, inplace: bool = True) -> "SysIdData":
        """
        Remove the mean from all series and store the means for later unscaling.
        """
        target = self if inplace else copy.deepcopy(self)

        for k, v in target:
            mu = np.mean(v)
            target.series[k] = v - mu
            # Store the mean (additively: if already centered, add to existing)
            target.means[k] = target.means.get(k, 0.0) + mu

        return target

    def standardize(self, inplace: bool = True) -> "SysIdData":
        """
        Remove mean and scale to unit variance (Z-score normalization).

        Stores both means and standard deviations for later unscaling.
        """
        target = self if inplace else copy.deepcopy(self)

        # First center (this fills target.means)
        target.center(inplace=True)

        # Then scale to unit variance
        for k, v in target:
            sigma = np.std(v)
            if sigma < 1e-12:  # Protect against division by zero for constant signals
                sigma = 1.0
                warnings.warn(f"Series '{k}' is constant. Skipping scaling.", stacklevel=2)

            target.series[k] = v / sigma
            # Store the scaling factor (multiplicatively)
            target.stds[k] = target.stds.get(k, 1.0) * sigma

        return target

    def detrend(self, method: str = "linear", inplace: bool = True) -> "SysIdData":
        """
        Remove trends from all series.

        Args:
            method: The detrending method to use:
                - 'linear': Remove linear trend using least-squares fit (default)
                - 'constant': Remove constant trend (mean subtraction). Equivalent to center().
                - 'standardized': Z-standardization (mean=0, std=1). Equivalent to standardize().
            inplace: If True, modify in-place. If False, return a copy.

        Returns:
            SysIdData: The detrended object (self if inplace=True, copy if inplace=False).

        Note:
            For 'constant' and 'standardized' methods, the scaling state (means/stds) is stored
            and can be reversed with unscale(). For 'linear' method, trend parameters (slope, intercept)
            are stored in trend_slopes and trend_intercepts and can also be reversed with unscale().
        """
        target = self if inplace else copy.deepcopy(self)

        if method == "constant":
            return target.center(inplace=True)
        elif method == "standardized":
            return target.standardize(inplace=True)
        elif method == "linear":
            # Use least-squares to fit and store trend parameters
            for k, v in target:
                t = target.time  # Use time vector for x-axis
                # Fit line: v = slope * t + intercept
                A = np.vstack([t, np.ones(len(t))]).T
                slope, intercept = np.linalg.lstsq(A, v, rcond=None)[0]
                # Store parameters for later reversal
                target.trend_slopes[k] = float(slope)
                target.trend_intercepts[k] = float(intercept)
                # Remove trend
                target.series[k] = v - (slope * t + intercept)
            return target
        else:
            raise ValueError(
                f"Invalid detrend method '{method}'. Must be one of: 'linear', 'constant', 'standardized'."
            )

    def unscale(self, inplace: bool = True) -> "SysIdData":
        """
        Reverse center(), standardize(), and detrend() transformations (back to physical units).

        The reverse operations are applied in the opposite order they were applied:
        1. Reverse detrending (add back linear trends)
        2. Reverse scaling (multiply by stored stds)
        3. Reverse centering (add back stored means)
        """
        target = self if inplace else copy.deepcopy(self)

        # 1. Reverse detrending (add back linear trends)
        if target.trend_slopes or target.trend_intercepts:
            for k, v in target:
                if k in target.trend_slopes and k in target.trend_intercepts:
                    t = target.time
                    trend = target.trend_slopes[k] * t + target.trend_intercepts[k]
                    target.series[k] = v + trend
            target.trend_slopes.clear()  # Reset
            target.trend_intercepts.clear()  # Reset

        # 2. Reverse scaling (multiply by stored stds)
        if target.stds:
            for k, v in target:
                if k in target.stds:
                    target.series[k] = v * target.stds[k]
            target.stds.clear()  # Reset

        # 3. Reverse centering (add back stored means)
        if target.means:
            for k, v in target:
                if k in target.means:
                    target.series[k] = v + target.means[k]
            target.means.clear()  # Reset

        return target

    def apply_scaling_from(self, source: "SysIdData", inplace: bool = True) -> "SysIdData":
        """
        Apply the scaling (means/stds) from another dataset to this one.

        Important for train/test splits: test data should be scaled using training statistics.
        """
        target = self if inplace else copy.deepcopy(self)

        # Apply means from source
        for k, v in target:
            if k in source.means:
                mu = source.means[k]
                target.series[k] = v - mu
                target.means[k] = target.means.get(k, 0.0) + mu

        # Apply stds from source
        for k, _ in target:
            if k in source.stds:
                sigma = source.stds[k]
                target.series[k] = target.series[k] / sigma
                target.stds[k] = target.stds.get(k, 1.0) * sigma

        return target

    def apply_detrend_from(self, source: "SysIdData", inplace: bool = True) -> "SysIdData":
        """
        Apply the detrending from another dataset to this one.

        Important for train/test splits: test data should be detrended using training trend parameters.
        Only applies linear detrending (trend_slopes and trend_intercepts).
        """
        target = self if inplace else copy.deepcopy(self)

        # Apply linear trend parameters from source
        for k, v in target:
            if k in source.trend_slopes and k in source.trend_intercepts:
                t = target.time
                slope = source.trend_slopes[k]
                intercept = source.trend_intercepts[k]
                trend = slope * t + intercept
                target.series[k] = v - trend
                # Store the applied parameters
                target.trend_slopes[k] = slope
                target.trend_intercepts[k] = intercept

        return target

    def crop(self, start: Optional[int] = None, end: Optional[int] = None, inplace: bool = True) -> "SysIdData":
        """
        Crop the data to a subset of samples.

        Args:
            start: Start index (default: 0).
            end: End index (default: N, exclusive).
            inplace: If True, modify in-place. If False, return a copy.

        Returns:
            SysIdData: The cropped object (self if inplace=True, copy if inplace=False).
        """
        start = start or 0
        end = end or self.N

        target = self if inplace else copy.deepcopy(self)

        # Update time vector
        if target.t is not None:
            # Non-equidistant: crop time vector and update t_start
            target.t = target.t[start:end]
            if len(target.t) > 0:
                target.t_start = float(target.t[0])
        else:
            # Equidistant: update t_start based on the number of skipped samples
            target.t_start += target.Ts * start

        # Crop all series
        for k, v in target:
            target.series[k] = v[start:end]

        return target

    def split(
        self, proportion: Optional[float] = None, sample: Optional[int] = None
    ) -> tuple["SysIdData", "SysIdData"]:
        """
        Split the data into two sets.

        Args:
            proportion: Ratio of the first set (0 to 1). Default 0.5.
            sample: Index to split at. Overrides proportion.

        Returns:
            A tuple containing two SysIdData objects.
        """
        if not sample and not proportion:
            proportion = 0.5

        if sample is None and proportion is not None:
            sample = int(round(self.N * proportion))

        # print(f"Splitting at {sample}")

        d1 = copy.deepcopy(self)
        d1.crop(end=sample, inplace=True)
        d2 = copy.deepcopy(self)
        d2.crop(start=sample, inplace=True)

        return d1, d2

    def resample(self, factor: float, inplace: bool = True) -> "SysIdData":
        """
        Resample the data using Fourier method (preserves frequency content).

        Args:
            factor: Resampling factor. >1 upsamples, <1 downsamples.
        """
        target = self if inplace else copy.deepcopy(self)
        N_new = int(target.N * factor)
        for k, v in target:
            target.series[k] = scipy.signal.resample(v, N_new)

        if target.t is not None:
            target.t = scipy.signal.resample(target.t, N_new)
        else:
            target.Ts = target.Ts / factor

        return target

    def downsample(self, q: int, inplace: bool = True) -> "SysIdData":
        """
        Downsample the data by an integer factor.

        Applies an anti-aliasing filter (Chebyshev type I) before downsampling.

        Args:
            q: Downsampling factor.
        """
        target = self if inplace else copy.deepcopy(self)
        if q < 1:
            raise ValueError(f"Downsampling factor must be >= 1, got {q}")

        for k, v in target:
            target.series[k] = scipy.signal.decimate(v, q)

        if target.Ts is not None:
            target.Ts *= q
        if target.t is not None:
            target.t = target.t[::q]

        return target

    def lowpass(self, order: int, corner_frequency: float, inplace: bool = True) -> "SysIdData":
        """
        Apply a Butterworth lowpass filter to all data series.

        Args:
            order: The order of the filter.
            corner_frequency: The corner frequency in Hz. Must be less than the Nyquist frequency.
            inplace: If True, modify in-place. If False, return a copy.

        Returns:
            SysIdData: The filtered object (self if inplace=True, copy if inplace=False).

        Raises:
            ValueError: If Ts is not defined or corner_frequency >= Nyquist frequency.
        """
        target = self if inplace else copy.deepcopy(self)
        if target.Ts is None:
            raise ValueError("Sampling time 'Ts' is required for filtering.")

        # Calculate Nyquist frequency and validate corner_frequency
        nyquist = 1.0 / (2 * target.Ts)
        if corner_frequency >= nyquist:
            raise ValueError(
                f"Corner frequency ({corner_frequency:.2f} Hz) must be less than Nyquist frequency "
                f"({nyquist:.2f} Hz). Increase sampling rate (decrease Ts={target.Ts}) or use a lower corner frequency. "
                f"Required Ts < {1.0 / (2 * corner_frequency):.4f}s"
            )

        sos = scipy.signal.butter(order, corner_frequency, "low", analog=False, fs=1.0 / target.Ts, output="sos")
        for k, _ in target:
            target.series[k] = scipy.signal.sosfilt(sos, target.series[k])
        return target

    def differentiate(self, key: str, new_key: Optional[str] = None, inplace: bool = True) -> "SysIdData":
        """
        Compute time derivative of a series using central differences.

        Uses numpy.gradient with central differences (edge_order=2) which:
        - Preserves array length (unlike np.diff which shortens by 1)
        - Uses second-order accurate central differences for interior points
        - Uses second-order forward/backward differences at boundaries

        Args:
            key: Name of the series to differentiate (e.g., "y" for position).
            new_key: Name for the derivative series. Defaults to "d{key}" (e.g., "dy" for "y").
            inplace: If True, modify in-place. If False, return a copy.

        Returns:
            SysIdData: The object with the derivative added (self if inplace=True, copy if inplace=False).

        Example:
            >>> data = SysIdData(Ts=0.01, position=np.array([0, 1, 2, 3, 4]))
            >>> data.differentiate("position", "velocity", inplace=True)
            >>> # Now data has "position" and "velocity" series
        """
        target = self if inplace else copy.deepcopy(self)

        if key not in target.series:
            raise ValueError(f"Series '{key}' not found in dataset.")

        # Determine the spacing for gradient calculation
        # For equidistant data, use Ts; for non-equidistant, use the time vector
        if target.Ts is not None:
            dt_arg = target.Ts
        else:
            dt_arg = target.time

        # Compute derivative using central differences
        y_dot = np.gradient(target.series[key], dt_arg, edge_order=2)

        # Determine output key name
        dest_key = new_key if new_key else f"d{key}"
        target.series[dest_key] = y_dot

        return target

    def plot(self) -> None:
        """Plot the data."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install it with 'pip install llsi[plot]'."
            ) from None

        t = self.time
        n_series = len(self.series)

        fig, axes = plt.subplots(n_series, 1, sharex=True, figsize=(10, 2 * n_series))
        if n_series == 1:
            axes = [axes]

        for ax, (key, val) in zip(axes, self):
            ax.plot(t, val, label=key)
            ax.legend()
            ax.grid(True)

        plt.xlabel("Time")
        plt.show()

    @staticmethod
    def generate_prbs(N: int, Ts: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a Pseudo-Random Binary Sequence (PRBS).

        Args:
            N: Number of samples.
            Ts: Sampling time.
            seed: Random seed.

        Returns:
            tuple of (time vector, PRBS signal).
        """
        t = np.linspace(0, Ts * N, num=N, endpoint=False)
        u = _math.generate_prbs_sequence(N, seed)
        return t, u

    def to_pandas(self) -> Any:
        """
        Convert to pandas DataFrame.

        Returns:
            pandas.DataFrame: The data as a DataFrame.
        """
        try:
            from .pandas_utils import to_pandas as _to_pandas
        except ImportError:
            raise ImportError("pandas is required for this method. Install it with 'pip install llsi[data]'.") from None

        return _to_pandas(self)

    @classmethod
    def from_pandas(cls, df, time_col=None, Ts=None):
        """
        Create SysIdData from pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing the data.
        time_col : str, optional
            Name of the column to use as time. If None, the index is used.
        Ts : float, optional
            Sampling time. If None, it is inferred from the time index if possible.
        """
        try:
            from .pandas_utils import from_pandas as _from_pandas
        except ImportError:
            raise ImportError("pandas is required for this method. Install it with 'pip install llsi[data]'.") from None

        return _from_pandas(df, time_col=time_col, Ts=Ts)

    @classmethod
    def from_logfile(
        cls,
        path,
        time_col="datetime",
        value_col="temperature",
        pivot_col="property_name",
        datetime_format=None,
        sep=",",
        **kwargs,
    ):
        """
        Loads a logfile, pivots it, and automatically regularizes the time grid using the
        internal equidistant() method.

        Parameters
        ----------
        path : str
            Path to the CSV logfile.
        time_col : str
            Column name containing datetime information.
        value_col : str
            Column name containing the measured value.
        pivot_col : str
            Column name that defines different signals.
        datetime_format : str
            Format string for faster datetime parsing.
        sep : str
            CSV separator.
        **kwargs
            Additional arguments passed to pd.read_csv.
        """
        try:
            from .pandas_utils import from_logfile as _from_logfile
        except ImportError:
            raise ImportError("pandas is required for this method. Install it with 'pip install llsi[data]'.") from None

        return _from_logfile(
            path,
            time_col=time_col,
            value_col=value_col,
            pivot_col=pivot_col,
            datetime_format=datetime_format,
            sep=sep,
            **kwargs,
        )
