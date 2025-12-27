#!/usr/bin/env python3
"""
Created on Sun Apr  4 19:40:17 2021

@author: armin
"""

import copy
import logging

import numpy as np
import scipy.interpolate
import scipy.signal


class SysIdData:
    def __init__(self, t=None, Ts=None, t_start=None, **kwargs):
        self.N = None
        self.series = {}
        self.add_series(**kwargs)
        self.t = t
        self.Ts = Ts

        if self.Ts is None and self.t is None:
            raise ValueError(
                "No time specified. Use eiter keyword t to give a series or Ts to "
                + "give a scalar for equidistant time series."
            )

        if self.t is not None:
            self.t_start = t[0]
        else:
            if t_start is None:
                self.t_start = 0.0
            else:
                self.t_start = t_start

        self.logger = logging.getLogger(__name__)

    def __getitem__(self, key):
        return self.series[key]

    def add_series(self, **kwargs):
        for key, val in kwargs.items():
            s = np.array(val).ravel()
            self.series[key] = s

            if self.N is None:
                self.N = s.shape[0]
            else:
                if self.N != s.shape[0]:
                    raise ValueError(
                        f"length of vector to add ({s.shape[0]}) " + "does not match length of time series ({self.N})"
                    )

    def remove(self, key):
        del self.series[key]

    def time(self):
        if self.t is not None:
            return self.t
        else:
            t_start = self.t_start
            t_end = t_start + (self.N - 1) * self.Ts
            return np.linspace(t_start, t_end, self.N)

    def equidistant(self, N=None):
        if N is None:
            N = self.N

        if N < self.N:
            self.logger.warning("Downsampling without filter!")

        t_ = self.time()

        t_start = t_[0]
        t_end = t_[-1]

        t = np.linspace(t_start, t_end, N)
        for key, val in self.series.items():
            f = scipy.interpolate.interp1d(self.t, val)
            self.series[key] = f(t)

        self.N = N
        self.Ts = (t_end - t_start) / (self.N - 1)
        self.t = None

    def center(self):
        for key, val in self.series.items():
            self.series[key] -= np.mean(val)

    def crop(self, start=None, end=None):
        if self.t is not None:
            self.t = self.t[start:end]
        else:
            if start:
                self.t_start += self.Ts * start

        for key, val in self.series.items():
            self.series[key] = val[start:end]
            self.N = self.series[key].shape[0]

    def downsample(self, q):
        for key, val in self.series.items():
            self.series[key] = scipy.signal.decimate(val, q)
            self.N = self.series[key].shape[0]
            self.Ts *= q

    def lowpass(self, order, corner_frequency):
        sos = scipy.signal.butter(order, corner_frequency, "low", analog=False, fs=1.0 / self.Ts, output="sos")

        for key in self.series.keys():
            self.series[key] = scipy.signal.sosfilt(sos, self.series[key])

    def split(self, proportion=None, sample=None):
        if not sample and not proportion:
            proportion = 0.5

        if proportion:
            sample = int(round(self.N * proportion))

        print(f"Splitting at {sample}")

        data1 = copy.deepcopy(self)  # .crop(start=0,end=sample)
        data1.crop(end=sample)
        data2 = copy.deepcopy(self)  # .crop(start=sample)
        data2.crop(start=sample)

        return data1, data2

    @staticmethod
    def generate_prbs(N, Ts, seed=42):
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
    def prbs31(code):
        for _ in range(32):
            next_bit = ~((code >> 30) ^ (code >> 27)) & 0x01
            code = ((code << 1) | next_bit) & 0xFFFFFFFF
        return code

    @staticmethod
    def prbs31_fast(code):
        next_code = ~((code << 1) ^ (code << 4)) & 0xFFFFFFF0
        next_code |= ~(((code << 1 & 0x0E) | (next_code >> 31 & 0x01)) ^ (next_code >> 28)) & 0x0000000F
        return next_code

    def to_pandas(self):
        """
        Convert to pandas DataFrame.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method") from None

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
