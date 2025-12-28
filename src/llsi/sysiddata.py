"""
Data container for system identification.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.interpolate
import scipy.signal


class SysIdData:
    """
    Container for time-series data used in system identification.
    
    Stores multiple time series (input/output channels) sharing a common time axis.
    Supports both equidistant and non-equidistant time sampling.
    """

    def __init__(
        self,
        t: Optional[np.ndarray] = None,
        Ts: Optional[float] = None,
        t_start: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Initialize SysIdData.

        Args:
            t: Time vector (for non-equidistant data).
            Ts: Sampling time (for equidistant data).
            t_start: Start time (for equidistant data). Default is 0.0.
            **kwargs: Time series data as keyword arguments (name=data).
        """
        self.N: Optional[int] = None
        self.series: Dict[str, np.ndarray] = {}
        self.add_series(**kwargs)
        self.t = t
        self.Ts = Ts

        if self.Ts is None and self.t is None:
            raise ValueError(
                "No time specified. Use either keyword 't' to give a time vector or 'Ts' to "
                "give a scalar for equidistant time series."
            )

        if self.t is not None:
            self.t_start = t[0]
        else:
            if t_start is None:
                self.t_start = 0.0
            else:
                self.t_start = t_start

        self.logger = logging.getLogger(__name__)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.series[key]

    def add_series(self, **kwargs: Any) -> None:
        """
        Add time series to the dataset.

        Args:
            **kwargs: Time series data as keyword arguments (name=data).
        """
        for key, val in kwargs.items():
            s = np.array(val).ravel()
            self.series[key] = s

            if self.N is None:
                self.N = s.shape[0]
            else:
                if self.N != s.shape[0]:
                    raise ValueError(
                        f"Length of vector to add ({s.shape[0]}) "
                        f"does not match length of time series ({self.N})"
                    )

    def remove(self, key: str) -> None:
        """Remove a time series."""
        del self.series[key]

    def time(self) -> np.ndarray:
        """
        Get the time vector.

        Returns:
            The time vector.
        """
        if self.t is not None:
            return self.t
        else:
            if self.N is None:
                return np.array([])
            t_start = self.t_start
            return t_start + np.arange(self.N) * self.Ts

    def equidistant(self, N: Optional[int] = None) -> None:
        """
        Resample data to be equidistant.
        
        Modifies the object in-place.

        Args:
            N: Number of points for the new grid. If None, keeps current N.
        """
        if self.t is None:
            # Already equidistant
            if N is not None and N != self.N:
                # Resample equidistant data
                pass 
            else:
                return

        if N is None:
            N = self.N

        if N < self.N:
            self.logger.warning("Downsampling without filter! Aliasing may occur.")

        t_current = self.time()
        t_start = t_current[0]
        t_end = t_current[-1]

        t_new = np.linspace(t_start, t_end, N)
        
        for key, val in self.series.items():
            f = scipy.interpolate.interp1d(t_current, val, kind='linear', fill_value="extrapolate")
            self.series[key] = f(t_new)

        self.N = N
        self.Ts = (t_end - t_start) / (self.N - 1) if self.N > 1 else 0.0
        self.t = None # Now it is equidistant

    def center(self) -> None:
        """Remove the mean from all series."""
        for key, val in self.series.items():
            self.series[key] -= np.mean(val)

    def crop(self, start: Optional[int] = None, end: Optional[int] = None) -> None:
        """
        Crop the data.

        Args:
            start: Start index.
            end: End index.
        """
        if self.t is not None:
            self.t = self.t[start:end]
        else:
            if start:
                self.t_start += self.Ts * start

        for key, val in self.series.items():
            self.series[key] = val[start:end]
            
        if self.series:
            self.N = next(iter(self.series.values())).shape[0]
        else:
            self.N = 0

    def split(self, proportion: Optional[float] = None, sample: Optional[int] = None) -> Tuple['SysIdData', 'SysIdData']:
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
        d1.crop(end=sample)
        d2 = copy.deepcopy(self)
        d2.crop(start=sample)

        return d1, d2

    def resample(self, factor: float) -> None:
        """
        Resample the data.

        Args:
            factor: Resampling factor. >1 upsamples, <1 downsamples.
        """
        N_new = int(self.N * factor)
        
        for key, val in self.series.items():
            self.series[key] = scipy.signal.resample(val, N_new)
            
        if self.t is not None:
             self.t = scipy.signal.resample(self.t, N_new)
        else:
             self.Ts = self.Ts / factor
             
        self.N = N_new

    def downsample(self, q: int) -> None:
        """
        Downsample the data by an integer factor.

        Args:
            q: Downsampling factor.
        """
        for key, val in self.series.items():
            self.series[key] = scipy.signal.decimate(val, q)
            
        if self.series:
            self.N = next(iter(self.series.values())).shape[0]
            
        if self.Ts is not None:
            self.Ts *= q
        if self.t is not None:
            self.t = self.t[::q]

    def lowpass(self, order: int, corner_frequency: float) -> None:
        """
        Apply a lowpass filter to all series.

        Args:
            order: Filter order.
            corner_frequency: Corner frequency in Hz.
        """
        sos = scipy.signal.butter(order, corner_frequency, "low", analog=False, fs=1.0 / self.Ts, output="sos")

        for key in self.series.keys():
            self.series[key] = scipy.signal.sosfilt(sos, self.series[key])

    def plot(self) -> None:
        """Plot the data."""
        import matplotlib.pyplot as plt

        t = self.time()
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

        code = seed
        i = 0
        while i < N:
            # generate integer
            code = SysIdData.prbs31(code)
            for s in f"{code:b}":
                if i >= N:
                    break
                u[i] = float(s)
                i += 1
        return t, u

    @staticmethod
    def prbs31(code: int) -> int:
        """PRBS31 generator step."""
        for _ in range(32):
            next_bit = ~((code >> 30) ^ (code >> 27)) & 0x01
            code = ((code << 1) | next_bit) & 0xFFFFFFFF
        return code

    @staticmethod
    def prbs31_fast(code: int) -> int:
        """Fast PRBS31 generator step."""
        next_code = ~((code << 1) ^ (code << 4)) & 0xFFFFFFF0
        next_code |= ~(((code << 1 & 0x0E) | (next_code >> 31 & 0x01)) ^ (next_code >> 28)) & 0x0000000F
        return next_code

    def to_pandas(self) -> Any:
        """
        Convert to pandas DataFrame.
        
        Returns:
            pandas.DataFrame: The data as a DataFrame.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method") from None
            
        df = pd.DataFrame(self.series)
        df.index = self.time()
        return df

        index = self.time()
        return pd.DataFrame(self.series, index=index)

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
            raise ImportError("pandas is required for this method") from None

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
