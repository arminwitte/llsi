# SysIdData Code Review & Improvement Suggestions

## Overview
The `SysIdData` class is a well-structured, feature-rich data container for system identification. It has excellent method chaining, comprehensive functionality, and solid documentation. Below are recommendations for further improvement.

---

## 1. 🔴 Critical Issues

### 1.1 Input Validation in `add_series()`
**Problem**: The method doesn't validate for empty arrays or NaN values.

**Current Code**:
```python
def add_series(self, **kwargs: Any) -> "SysIdData":
    for key, val in kwargs.items():
        s = np.atleast_1d(val).ravel()
        if self.series and s.shape[0] != self.N:
            raise ValueError(...)
        self.series[key] = np.asarray(s)
    return self
```

**Issues**:
- No check for all-NaN arrays
- No check for non-finite values (inf, nan)
- Silent overwrite if adding duplicate keys

**Suggestion**:
```python
def add_series(self, **kwargs: Any) -> "SysIdData":
    for key, val in kwargs.items():
        s = np.atleast_1d(val).ravel()
        
        # Validate length
        if self.series and s.shape[0] != self.N:
            raise ValueError(
                f"Length of vector to add ({s.shape[0]}) does not match existing series length ({self.N})"
            )
        
        # Check for all-NaN or all-inf
        if len(s) > 0 and np.all(~np.isfinite(s)):
            raise ValueError(f"Series '{key}' contains only NaN or infinite values.")
        
        # Warn on duplicate keys
        if key in self.series:
            self.logger.warning(f"Series '{key}' already exists. Overwriting.")
        
        self.series[key] = np.asarray(s, dtype=float)
    return self
```

### 1.2 Missing Input Validation in `__init__`
**Problem**: No validation that Ts is positive or that t is strictly increasing.

**Suggestion**:
```python
def __post_init__(self) -> None:
    self.logger = logging.getLogger(__name__)
    
    # Validate Ts
    if self.Ts is not None and self.Ts <= 0:
        raise ValueError(f"Sampling time Ts must be positive, got {self.Ts}")
    
    # Validate t is strictly increasing
    if self.t is not None and len(self.t) > 1:
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("Time vector t must be strictly increasing")
    
    # Ensure series arrays are numpy arrays
    for k, v in list(self.series.items()):
        self.series[k] = np.asarray(v).ravel()
    
    if self.Ts is None and self.t is None:
        raise ValueError("Either 't' (time vector) or 'Ts' (sampling time) must be provided.")
```

### 1.3 Potential Issue in `equidistant()` with Invalid Methods
**Problem**: If an invalid interpolation method is passed, `scipy.interpolate.interp1d` will fail with unclear error.

**Suggestion**: Add validation:
```python
def equidistant(self, N: Optional[int] = None, inplace: bool = True, 
                method: Union[str, Dict[str, str]] = "linear") -> "SysIdData":
    # ... existing code ...
    
    # Validate interpolation methods
    valid_methods = {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'}
    
    if isinstance(method, str):
        if method not in valid_methods:
            raise ValueError(f"Invalid interpolation method '{method}'. "
                           f"Must be one of {valid_methods}")
    else:
        for key, meth in method.items():
            if meth not in valid_methods:
                raise ValueError(f"Invalid interpolation method '{meth}' for series '{key}'")
```

---

## 2. 🟡 Design Improvements

### 2.1 Inconsistent Docstring Formats
**Problem**: Mix of NumPy-style and Google-style docstrings.

**Current inconsistencies**:
- `from_pandas()`: Uses NumPy style (Parameters, Returns)
- `from_logfile()`: Uses NumPy style (Parameters)
- `differentiate()`: Uses Google style (Args, Returns)

**Suggestion**: Standardize on Google-style docstrings throughout for consistency.

### 2.2 No `__len__` Method
**Problem**: Users can't do `len(data)` - must use `data.N`

**Suggestion**:
```python
def __len__(self) -> int:
    """Return number of samples."""
    return self.N
```

### 2.3 No `__contains__` Method
**Problem**: Users can't do `"u" in data` - must use `"u" in data.series`

**Suggestion**:
```python
def __contains__(self, key: str) -> bool:
    """Check if a series exists in the dataset."""
    return key in self.series
```

### 2.4 Missing Context Manager Support
**Problem**: Can't use `with` statement for temporary copies

**Suggestion**:
```python
def __enter__(self) -> "SysIdData":
    """Context manager entry."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Context manager exit."""
    pass

# Usage:
# with data.copy() as d_temp:
#     d_temp.lowpass(...)
#     # Process d_temp without modifying original
```

### 2.5 No `copy()` Method
**Problem**: Must use `copy.deepcopy(data)` - not intuitive

**Suggestion**:
```python
def copy(self) -> "SysIdData":
    """Return a deep copy of this SysIdData object."""
    return copy.deepcopy(self)
```

### 2.6 No Series Iteration
**Problem**: Can't iterate series easily

**Suggestion**:
```python
def __iter__(self):
    """Iterate over (key, array) tuples."""
    return iter(self.series.items())
```

### 2.7 No Slicing Support (`__getitem__` with slice)
**Problem**: `data[0:10]` doesn't work - must use `data.crop(0, 10)`

**Suggestion**:
```python
def __getitem__(self, key: Union[str, slice, int]) -> Union[np.ndarray, "SysIdData"]:
    """
    Get a series by name or slice data by index.
    
    Examples:
        data["u"]  # Get series named "u"
        data[0:10]  # Crop to first 10 samples
        data[:5]  # First 5 samples
    """
    if isinstance(key, str):
        return self.series[key]
    elif isinstance(key, (slice, int)):
        return self.crop(start=key.start if isinstance(key, slice) else key,
                        end=key.stop if isinstance(key, slice) else key+1,
                        inplace=False)
    else:
        raise TypeError(f"Invalid index type: {type(key)}")
```

---

## 3. 🟡 Performance & Edge Cases

### 3.1 `crop()` Doesn't Update `t_start` Correctly for Non-Equidistant Data
**Problem**: When cropping non-equidistant data, `t_start` isn't updated.

**Current Code**:
```python
def crop(self, start: Optional[int] = None, end: Optional[int] = None, inplace: bool = True) -> "SysIdData":
    target = self if inplace else copy.deepcopy(self)
    if target.t is not None:
        target.t = target.t[start:end]
    else:
        if start:
            target.t_start += target.Ts * start
    # ...
```

**Issue**: If `target.t is not None`, the code should update the logic.

**Suggestion**:
```python
def crop(self, start: Optional[int] = None, end: Optional[int] = None, inplace: bool = True) -> "SysIdData":
    start = start or 0
    end = end or self.N
    
    target = self if inplace else copy.deepcopy(self)
    
    if target.t is not None:
        target.t = target.t[start:end]
        if len(target.t) > 0:
            target.t_start = float(target.t[0])
    else:
        target.t_start += target.Ts * start
    
    for k, v in list(target.series.items()):
        target.series[k] = v[start:end]
    
    return target
```

### 3.2 `resample()` Uses Default Fourier Resampling
**Problem**: `scipy.signal.resample` uses FFT (frequency-domain), which:
- May introduce artifacts at boundaries
- Assumes periodic continuation
- Not ideal for control signals with discontinuities

**Suggestion**: Add optional method parameter:
```python
def resample(self, factor: float, method: str = "fft", inplace: bool = True) -> "SysIdData":
    """
    Resample the data.
    
    Args:
        factor: Resampling factor. >1 upsamples, <1 downsamples.
        method: 'fft' (default, smooth) or 'linear' (piecewise), or 'cubic'.
    """
    target = self if inplace else copy.deepcopy(self)
    N_new = int(target.N * factor)
    
    if method == "fft":
        for k, v in list(target.series.items()):
            target.series[k] = scipy.signal.resample(v, N_new)
    else:
        # Use equidistant() for flexible interpolation
        return target.equidistant(N=N_new, method=method, inplace=inplace)
    
    if target.t is not None:
        target.t = scipy.signal.resample(target.t, N_new)
    else:
        target.Ts = target.Ts / factor
    
    return target
```

### 3.3 `downsample()` Doesn't Warn About Aliasing
**Problem**: Integer downsampling can introduce aliasing without a lowpass filter first.

**Suggestion**:
```python
def downsample(self, q: int, inplace: bool = True, warn: bool = True) -> "SysIdData":
    """
    Downsample by integer factor.
    
    Warning: Applies lowpass filter internally to prevent aliasing.
    """
    if q < 1:
        raise ValueError(f"Downsampling factor must be >= 1, got {q}")
    
    target = self if inplace else copy.deepcopy(self)
    
    # Apply anti-aliasing filter first (scipy.signal.decimate does this internally)
    for k, v in list(target.series.items()):
        target.series[k] = scipy.signal.decimate(v, q)
    
    if target.Ts is not None:
        target.Ts *= q
    if target.t is not None:
        target.t = target.t[::q]
    
    return target
```

### 3.4 `lowpass()` Doesn't Validate corner_frequency
**Problem**: No check that corner_frequency < Nyquist frequency.

**Suggestion**:
```python
def lowpass(self, order: int, corner_frequency: float, inplace: bool = True) -> "SysIdData":
    target = self if inplace else copy.deepcopy(self)
    if target.Ts is None:
        raise ValueError("Sampling time 'Ts' is required for filtering.")
    
    nyquist = 1.0 / (2 * target.Ts)
    if corner_frequency >= nyquist:
        raise ValueError(
            f"Corner frequency ({corner_frequency} Hz) must be < Nyquist "
            f"({nyquist:.2f} Hz). Use Ts={target.Ts} (or lower Ts to increase Nyquist)."
        )
    
    sos = scipy.signal.butter(
        order, corner_frequency, "low", 
        analog=False, fs=1.0 / target.Ts, 
        output="sos"
    )
    for k in list(target.series.keys()):
        target.series[k] = scipy.signal.sosfilt(sos, target.series[k])
    return target
```

### 3.5 `differentiate()` Doesn't Handle Series With NaNs
**Problem**: `np.gradient` propagates NaNs, which can corrupt entire signal.

**Suggestion**:
```python
def differentiate(self, key: str, new_key: Optional[str] = None, inplace: bool = True) -> "SysIdData":
    target = self if inplace else copy.deepcopy(self)
    
    if key not in target.series:
        raise ValueError(f"Series '{key}' not found in dataset.")
    
    y = target.series[key]
    
    # Warn if series contains NaNs
    if np.any(np.isnan(y)):
        self.logger.warning(
            f"Series '{key}' contains {np.sum(np.isnan(y))} NaN values. "
            "Derivative will propagate NaNs."
        )
    
    if target.Ts is not None:
        dt_arg = target.Ts
    else:
        dt_arg = target.time
    
    y_dot = np.gradient(y, dt_arg, edge_order=2)
    dest_key = new_key if new_key else f"d{key}"
    target.series[dest_key] = y_dot
    
    return target
```

---

## 4. 📚 Documentation Improvements

### 4.1 No Module-Level Docstring Examples
**Suggestion**: Add usage examples in the class docstring:

```python
"""
Container for time-series data used in system identification.

Features:
    - Uses dataclasses for concise initialization
    - Supports equidistant and non-equidistant time sampling
    - Method chaining for fluent API design
    - Flexible data manipulation (crop, resample, filter, differentiate)

Examples:
    >>> # Create equidistant data
    >>> data = SysIdData(Ts=0.01, u=u_array, y=y_array)
    
    >>> # Create non-equidistant data from CSV
    >>> data = SysIdData.from_logfile("measurements.csv")
    
    >>> # Method chaining
    >>> data.equidistant(N=1000).lowpass(order=4, corner_frequency=10).plot()
    
    >>> # Access series
    >>> velocity = data["position"].differentiate("position", "velocity")
"""
```

### 4.2 Missing Examples in Method Docstrings
**Suggestion**: Add examples to complex methods like `split()`, `resample()`, `equidistant()`.

---

## 5. ✅ Strong Points (Keep These!)

1. **Excellent method chaining design** - All mutating methods return `self`
2. **Clear inplace/copy pattern** - Consistent across all methods
3. **Flexible equidistant() method** - Supports per-series interpolation
4. **Automatic Ts inference** - from_pandas and from_logfile infer Ts intelligently
5. **Non-equidistant data support** - Rare and valuable feature
6. **Comprehensive __repr__** - Jupyter-friendly output
7. **Good error messages** - Helpful validation with context

---

## 6. Quick Win Improvements (Easy to Implement)

Priority order:

1. **Add `copy()` method** - Users expect this
2. **Add `__len__()` and `__contains__()` methods** - Pythonic
3. **Validate `Ts > 0` in `__post_init__`** - Prevents silent bugs
4. **Validate interpolation method in `equidistant()`** - Clearer errors
5. **Add `__iter__()` for series** - Better ergonomics
6. **Standardize docstring format** - Consistency
7. **Enhance crop() for non-equidistant data** - Correctness
8. **Add corner_frequency validation in lowpass()** - Prevent silent errors

---

## Summary Table

| Issue | Severity | Effort | Impact |
|-------|----------|--------|--------|
| Validate Ts > 0 | Critical | Low | High |
| Validate method in equidistant() | Critical | Low | High |
| Add copy() method | Medium | Very Low | Medium |
| Add __len__/__contains__ | Low | Very Low | Low |
| Improve crop() for non-equidistant | Medium | Low | Medium |
| Standardize docstrings | Low | Medium | Low |
| Add corner_frequency validation | Medium | Low | High |

