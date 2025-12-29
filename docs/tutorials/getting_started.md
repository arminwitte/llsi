# Getting Started

## Installation

```bash
pip install llsi
```

## Basic Usage

```python
import llsi
import numpy as np

# Generate data
data = llsi.SysIdData(t=np.arange(100), u=np.random.randn(100), y=np.random.randn(100))

# Identify model
mod = llsi.sysid(data, 'y', 'u', method='n4sid')

# Plot
with llsi.Figure() as fig:
    fig.plot(mod)
```
