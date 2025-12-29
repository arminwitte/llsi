# llsi Documentation

Welcome to the documentation for **llsi** (Lightweight Linear System Identification).

## Overview

**llsi** is a Python library for identifying linear time-invariant (LTI) systems from data. It provides a simple, unified interface for various system identification algorithms.

Currently implemented methods:

*   **Subspace Identification**: `n4sid`, `po-moesp`
*   **Polynomial Models**: `arx` (AutoRegressive with eXogenous input)
*   **Optimization**: `pem` (Prediction Error Method), `oe` (Output Error)

## Installation

```bash
pip install llsi
```

## Quick Start

```python
import llsi
import numpy as np

# 1. Prepare Data
data = llsi.SysIdData(t=np.arange(100), u=np.random.randn(100), y=np.random.randn(100))

# 2. Identify Model
mod = llsi.sysid(data, 'y', 'u', method='n4sid')

# 3. Validate
with llsi.Figure() as fig:
    fig.plot(mod)
```

## Documentation Structure

*   [Tutorials](tutorials/getting_started.md): Step-by-step guides for using `llsi`.
*   [API Reference](api/ltimodel.md): Detailed documentation of classes and functions.

## Examples

For more interactive examples, check out the [notebooks](https://github.com/arminwitte/llsi/tree/main/notebooks) in the repository.
