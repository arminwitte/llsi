# Interpolation Methods in `equidistant()`

The `equidistant()` method supports configurable interpolation to handle different signal types appropriately in system identification workflows.

## Problem

Linear interpolation is suitable for continuous signals but introduces artifacts in step signals (like digital control outputs). Zero-Order Hold (ZOH, implemented as "previous" in scipy) is more appropriate for discrete-time control inputs.

## Solution

Use the `method` parameter to specify interpolation per-series:

### Example 1: Single Method for All Series

```python
from llsi import SysIdData
import numpy as np

# Create non-equidistant data
t = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
u = np.array([0.0, 1.0, 1.0, 0.5, 0.5])  # Control input (step function)
y = np.array([0.0, 0.2, 0.5, 0.7, 0.9])  # Physical output (continuous)

data = SysIdData(t=t, u=u, y=y)

# Linear interpolation (default, backward compatible)
data_linear = SysIdData(t=t, u=u, y=y)
data_linear.equidistant(N=100)  # Uses "linear" for all series

# Zero-Order Hold for all
data_zoh = SysIdData(t=t, u=u, y=y)
data_zoh.equidistant(N=100, method="previous")
```

### Example 2: Per-Series Methods (Recommended for System ID)

```python
# Control input uses ZOH, measurement uses linear interpolation
data = SysIdData(t=t, u=u, y=y)
data.equidistant(
    N=100,
    method={"u": "previous", "y": "linear"}  # Per-series specification
)
```

## Available Methods

The `method` parameter accepts any scipy.interpolate.interp1d `kind` argument:

- **"linear"** (default): Linear interpolation between points. Smooth but can introduce artifacts in step signals.
- **"previous"**: Zero-Order Hold (step). Preserves step structure for control inputs.
- **"cubic"**: Cubic spline interpolation. Smooth with better properties than linear.
- **"nearest"**: Nearest-neighbor interpolation.

## System Identification Best Practices

For mixed control/measurement data:

```python
# Typical system identification setup
data = SysIdData.from_logfile("experiment.csv", ...)
data.equidistant(
    N=1000,
    method={
        "u": "previous",  # Control inputs (step outputs from DACs)
        "y": "linear",    # Physical measurements
    }
)

# Now ready for algorithms (N4SID, PO_MOESP, etc.)
```

## Impact on Results

- **"previous" on control input**: Preserves step structure, more realistic for digital control
- **"linear" on measurements**: Smooth reconstruction of continuous signals
- This combination maintains signal integrity for better system identification accuracy

## Backward Compatibility

The default method is "linear", so existing code continues to work unchanged:

```python
data.equidistant(N=100)  # Still uses "linear" like before
```
