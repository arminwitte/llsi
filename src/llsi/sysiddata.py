"""
Data container for system identification.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

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
    - `series` stores named 1-D numpy arrays
    - `t` holds a non-equidistant time vector or None for equidistant data
    - `Ts` is the sampling time for equidistant data
    Methods that mutate return `self` to allow method chaining.
    """

    series: Dict[str, np.ndarray] = field(default_factory=dict)
    t: Optional[np.ndarray] = None
    Ts: Optional[float] = None
    t_start: float = 0.0

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
        # Call post-init validations and conversions
        self.__post_init__()

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        # Ensure series arrays are numpy arrays and set N
        for k, v in list(self.series.items()):
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

    def __getitem__(self, key: str) -> np.ndarray:
        return self.series[key]

    def add_series(self, **kwargs: Any) -> "SysIdData":
        """
        Add time series to the dataset.

        Args:
            **kwargs: Time series data as keyword arguments (name=data).
        """
        for key, val in kwargs.items():
            s = np.atleast_1d(val).ravel()
            if self.series and s.shape[0] != self.N:
                raise ValueError(
                    f"Length of vector to add ({s.shape[0]}) does not match existing series length ({self.N})"
                )
            self.series[key] = np.asarray(s)
        return self

    def remove(self, key: str) -> "SysIdData":
        """Remove a time series and return self for chaining."""
        del self.series[key]
        return self

    # `time` property implemented above

    def equidistant(self, N: Optional[int] = None, inplace: bool = True) -> "SysIdData":
        """
        Resample data to be equidistant.

        Modifies the object in-place.

        Args:
            N: Number of points for the new grid. If None, keeps current N.
        """
        target = self if inplace else copy.deepcopy(self)

        if N is None:
            N = target.N

        if N < target.N:
            target.logger.warning("Downsampling without filter! Aliasing may occur.")

        t_current = np.asarray(target.time)
        if t_current.size == 0:
            return target

        t_start = t_current[0]
        t_end = t_current[-1]
        t_new = np.linspace(t_start, t_end, N)

        if target.series:
            keys = list(target.series.keys())
            data_matrix = np.stack([target.series[k] for k in keys], axis=0)
            f = scipy.interpolate.interp1d(t_current, data_matrix, kind="linear", axis=1, fill_value="extrapolate")
            new_matrix = f(t_new)
            for i, k in enumerate(keys):
                target.series[k] = new_matrix[i, :]

        target.Ts = (t_end - t_start) / (N - 1) if N > 1 else 0.0
        target.t = None
        return target

    def center(self, inplace: bool = True) -> "SysIdData":
        """Remove the mean from all series. Returns self (or a copy if inplace=False)."""
        target = self if inplace else copy.deepcopy(self)
        for k, v in list(target.series.items()):
            target.series[k] = v - np.mean(v)
        return target

    def crop(self, start: Optional[int] = None, end: Optional[int] = None, inplace: bool = True) -> "SysIdData":
        """
        Crop the data.

        Args:
            start: Start index.
            end: End index.
        """
        target = self if inplace else copy.deepcopy(self)
        if target.t is not None:
            target.t = target.t[start:end]
        else:
            if start:
                target.t_start += target.Ts * start

        for k, v in list(target.series.items()):
            target.series[k] = v[start:end]

        return target

    def split(
        self, proportion: Optional[float] = None, sample: Optional[int] = None
    ) -> Tuple["SysIdData", "SysIdData"]:
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
        Resample the data.

        Args:
            factor: Resampling factor. >1 upsamples, <1 downsamples.
        """
        target = self if inplace else copy.deepcopy(self)
        N_new = int(target.N * factor)
        for k, v in list(target.series.items()):
            target.series[k] = scipy.signal.resample(v, N_new)

        if target.t is not None:
            target.t = scipy.signal.resample(target.t, N_new)
        else:
            target.Ts = target.Ts / factor

        return target

    def downsample(self, q: int, inplace: bool = True) -> "SysIdData":
        """
        Downsample the data by an integer factor.

        Args:
            q: Downsampling factor.
        """
        target = self if inplace else copy.deepcopy(self)
        for k, v in list(target.series.items()):
            target.series[k] = scipy.signal.decimate(v, q)

        if target.Ts is not None:
            target.Ts *= q
        if target.t is not None:
            target.t = target.t[::q]

        return target

    def lowpass(self, order: int, corner_frequency: float, inplace: bool = True) -> "SysIdData":
        """
        Apply a Butterworth lowpass filter to all data series.

        This method modifies the data in-place.

        Args:
            order: The order of the filter.
            corner_frequency: The corner frequency in Hz.
        """
        target = self if inplace else copy.deepcopy(self)
        if target.Ts is None:
            raise ValueError("Sampling time 'Ts' is required for filtering.")
        sos = scipy.signal.butter(order, corner_frequency, "low", analog=False, fs=1.0 / target.Ts, output="sos")
        for k in list(target.series.keys()):
            target.series[k] = scipy.signal.sosfilt(sos, target.series[k])
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

        for ax, (key, val) in zip(axes, self.series.items()):
            ax.plot(t, val, label=key)
            ax.legend()
            ax.grid(True)

        plt.xlabel("Time")
        plt.show()

    @staticmethod
    def generate_prbs(N: int, Ts: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Pseudo-Random Binary Sequence (PRBS).

        Args:
            N: Number of samples.
            Ts: Sampling time.
            seed: Random seed.

        Returns:
            Tuple of (time vector, PRBS signal).
        """
        u = np.zeros((N,), dtype=float)
        t = np.linspace(0, Ts * N, num=N, endpoint=False)

        code = int(seed)
        i = 0
        while i < N:
            # Advance the PRBS state using the centralized helper
            code = _math.prbs31(code)
            bits = f"{code:b}"
            for s in bits:
                if i >= N:
                    break
                u[i] = float(int(s))
                i += 1
        return t, u

    @staticmethod
    def prbs31(code: int) -> int:
        """Delegate PRBS31 step to math.prbs31."""
        return int(_math.prbs31(int(code)))

    @staticmethod
    def prbs31_fast(code: int) -> int:
        """Delegate fast PRBS31 step to math.prbs31_fast."""
        return int(_math.prbs31_fast(int(code)))

    def to_pandas(self) -> Any:
        """
        Convert to pandas DataFrame.

        Returns:
            pandas.DataFrame: The data as a DataFrame.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method. Install it with 'pip install llsi[data]'.") from None

        df = pd.DataFrame(self.series)
        df.index = self.time
        return df

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
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method. Install it with 'pip install llsi[data]'.") from None

        if time_col:
            t_values = df[time_col].values
            data_df = df.drop(columns=[time_col])
        else:
            t_values = df.index.values
            data_df = df

        series_data = {col: data_df[col].values for col in data_df.columns}

        # Infer Ts if not provided
        t_start = None
        t_vec = None

        if Ts is None:
            # Check if t_values are numeric or datetime
            if pd.api.types.is_numeric_dtype(t_values):
                diffs = np.diff(t_values)
                if len(diffs) > 0 and np.allclose(diffs, diffs[0]):
                    Ts = float(diffs[0])
                    t_start = float(t_values[0])
                else:
                    t_vec = t_values
            elif pd.api.types.is_datetime64_any_dtype(t_values):
                # Convert to seconds relative to start
                t_start_timestamp = t_values[0]
                t_seconds = (t_values - t_start_timestamp) / np.timedelta64(1, "s")

                diffs = np.diff(t_seconds)
                if len(diffs) > 0 and np.allclose(diffs, diffs[0]):
                    Ts = float(diffs[0])
                    t_start = 0.0  # Relative time
                else:
                    t_vec = t_seconds
                    t_start = 0.0
            else:
                # Fallback, maybe just index
                t_vec = np.arange(len(t_values))
                Ts = 1.0
                t_start = 0.0
        else:
            # Ts provided
            if pd.api.types.is_numeric_dtype(t_values):
                t_start = float(t_values[0])
            else:
                t_start = 0.0

        return cls(t=t_vec, Ts=Ts, t_start=t_start, **series_data)

    @classmethod
    def from_logfile(
        cls,
        path,
        resample_rule=None,
        time_col="datetime",
        value_col="temperature",
        pivot_col="property_name",
        datetime_format=None,
        interpolate_method="linear",
        N: Optional[int] = None,
    ):
        """
        Read a temperature logfile CSV, pivot to a time-indexed DataFrame,
        resample to an equidistant grid and return a SysIdData instance.

        Parameters
        ----------
        path : str
            Path to the CSV logfile.
        resample_rule : str
            Pandas resample rule (e.g., '1H', '15T').
        time_col : str
            Column name containing datetime information.
        value_col : str
            Column name containing the measured value.
        pivot_col : str
            Column name that defines different signals (becomes columns).
        datetime_format : str or None
            Optional datetime format for faster parsing.
        interpolate_method : str
            Interpolation method passed to pandas.DataFrame.interpolate.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method. Install it with 'pip install llsi[data]'.") from None

        df = pd.read_csv(path)

        # Ensure datetime column exists and convert
        if time_col not in df.columns:
            raise KeyError(f"time column '{time_col}' not found in logfile")

        df[time_col] = pd.to_datetime(df[time_col], format=datetime_format)

        # Create pivot table: index=time, columns=pivot_col, values=value_col
        if pivot_col not in df.columns:
            raise KeyError(f"pivot column '{pivot_col}' not found in logfile")
        if value_col not in df.columns:
            raise KeyError(f"value column '{value_col}' not found in logfile")

        df_pivot = df.pivot(index=time_col, columns=pivot_col, values=value_col)

        # Resample to equidistant time grid and interpolate small gaps
        if resample_rule is not None:
            df_resampled = df_pivot.resample(resample_rule).mean()
            df_interp = df_resampled.interpolate(method=interpolate_method)
            # Drop rows that are still NaN (e.g., large gaps)
            df_clean = df_interp.dropna(how="any")
        else:
            df_clean = df_pivot
        if df_clean.shape[0] == 0:
            raise ValueError("No data left after resampling and interpolation. Check the resample_rule or input data.")

        # Build time vector in seconds relative to start
        index = df_clean.index
        t_seconds = (index - index[0]) / np.timedelta64(1, "s")

        series_data = {col: df_clean[col].values for col in df_clean.columns}

        obj = cls(t=t_seconds.values, Ts=None, t_start=0.0, **series_data)

        # If N is provided, return an equidistant copy with N points.
        if N is not None:
            return obj.equidistant(N=N, inplace=False)

        # Default: return equidistant data matching current sample count
        return obj.equidistant(N=obj.N, inplace=False)
