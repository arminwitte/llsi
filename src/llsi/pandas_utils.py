"""
Pandas utilities for SysIdData.

This module provides optional pandas integration for llsi.
All functions require pandas to be installed.
"""

from typing import Any, Optional

import numpy as np


def to_pandas(sysid_data) -> Any:
    """
    Convert SysIdData to pandas DataFrame.

    Parameters
    ----------
    sysid_data : SysIdData
        The system identification data to convert.

    Returns
    -------
    pandas.DataFrame
        The data as a DataFrame with time as index.
    """
    import pandas as pd

    df = pd.DataFrame(sysid_data.series)
    df.index = sysid_data.time
    return df


def from_pandas(df, time_col: Optional[str] = None, Ts: Optional[float] = None) -> Any:
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

    Returns
    -------
    SysIdData
        The system identification data object.
    """
    import pandas as pd

    from .sysiddata import SysIdData

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

    return SysIdData(t=t_vec, Ts=Ts, t_start=t_start, **series_data)


def from_logfile(
    path,
    time_col: str = "datetime",
    value_col: str = "temperature",
    pivot_col: str = "property_name",
    datetime_format: Optional[str] = None,
    sep: str = ",",
    **kwargs,
) -> Any:
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

    Returns
    -------
    SysIdData
        The loaded and processed system identification data.
    """
    import pandas as pd

    from .sysiddata import SysIdData

    # 1. Load Raw Data
    df = pd.read_csv(path, sep=sep, **kwargs)

    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found.")

    # Convert to datetime
    df[time_col] = pd.to_datetime(df[time_col], format=datetime_format)

    # 2. Pivot (Make wide)
    # We use pivot_table with 'first' to handle duplicates strictly,
    # or just pivot if we are sure data is unique per timestamp.
    # pivot_table is safer for dirty logs.
    if pivot_col:
        df_wide = df.pivot_table(index=time_col, columns=pivot_col, values=value_col, aggfunc="first")
    else:
        # If no pivot_col, assume each row is a different signal
        # This is a simple case where we just rename columns
        df_wide = df.rename(columns={value_col: df[value_col].name if hasattr(df[value_col], "name") else value_col})
        df_wide = df_wide.set_index(time_col)

    # 3. Drop rows with all NaN values (can happen with pivot_table)
    df_wide = df_wide.dropna(how="all")

    # 4. Convert to SysIdData
    data = SysIdData.from_pandas(df_wide)

    # 5. Regularize time grid
    data.equidistant()

    return data
