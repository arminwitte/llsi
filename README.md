# llsi
Lightweight Linear System Identification package.

`llsi` offers easy access to system identification algorithms in Python. It provides tools for identifying state-space models (*n4sid*, *PO-MOESP*) and transfer function models (*ARX*). Additionally, it supports prediction error methods (*PEM*) for output-error (*OE*) models or iterative improvement of state-space models.

To try it out online, you can use [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminwitte/llsi/HEAD?labpath=notebooks%2Fexample.ipynb).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![PyPI version](https://badge.fury.io/py/llsi.svg)](https://badge.fury.io/py/llsi)

## Installation

Install `llsi` using pip:

```bash
pip install llsi
```

To include optional dependencies like `scikit-learn` support:

```bash
pip install llsi[sklearn]
```

## Features

*   **Algorithms**: Subspace methods (N4SID, PO-MOESP), ARX, Output-Error (OE), and PEM.
*   **MATLAB-like API**: Familiar syntax for users migrating from MATLAB's System Identification Toolbox (`llsi.matlab`).
*   **Pandas Integration**: Seamlessly convert between `SysIdData` and pandas DataFrames.
*   **Scikit-learn Wrapper**: Use LTI models as scikit-learn estimators.
*   **Lightweight**: Core dependencies are just `numpy`, `scipy`, `pandas`, and `matplotlib`.

## Usage

### Basic Identification

1.  **Load data**:
    Start by loading your data. Here we use the heated wire dataset included in the repo.

    ```python
    import numpy as np
    import llsi

    # Load data (time, input Re, output Nu)
    d = np.load('data/heated_wire_data.npy')
    t, Re, Nu = d[:, 0], d[:, 1], d[:, 2]
    ```

2.  **Create a SysIdData object**:
    ```python
    data = llsi.SysIdData(t=t, Re=Re, Nu=Nu)
    ```

3.  **Preprocess**:
    Ensure equidistant time steps, remove transient start, and center the data.
    ```python
    data.equidistant()
    data.downsample(3)
    data.crop(start=100)
    data.center()
    ```

4.  **Identify a model**:
    Identify a state-space model with order 3 using the "PO-MOESP" algorithm.
    ```python
    mod = llsi.sysid(data, 'Nu', 'Re', (3,), method='po-moesp')
    ```

5.  **Export**:
    Convert to a `scipy.signal.StateSpace` object or transfer function.
    ```python
    ss = mod.to_ss()
    tf = mod.to_tf(continuous=True)
    ```

### Pandas Integration

```python
import pandas as pd
from llsi import SysIdData

# Create from DataFrame (infers sampling time from index)
df = pd.read_csv("my_data.csv", index_col="timestamp", parse_dates=True)
data = SysIdData.from_pandas(df)

# Export back to DataFrame
df_new = data.to_pandas()
```

### MATLAB-like API

If you are familiar with MATLAB's System Identification Toolbox, you can use the `llsi.matlab` module:

```python
from llsi.matlab import iddata, n4sid, compare

# Create iddata object
data = iddata(y, u, Ts)

# Identify model
sys = n4sid(data, 3)

# Compare
compare(data, sys)
```

### Plotting

Use the `llsi.Figure` context manager for quick visualizations:

```python
with llsi.Figure() as fig:
    fig.plot(mod, 'impulse')
    fig.plot(mod, 'step')
```

## Contribution
Thank you for considering to contribute. Any exchange and help is welcome. However, I have to ask you to be patient with me responding.
