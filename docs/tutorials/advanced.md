# Advanced Usage

## Residual Analysis

```python
with llsi.Figure() as fig:
    fig.plot({"mod": mod, "data": val_data}, "residuals")
```

## Input Delay

```python
mod = llsi.sysid(data, 'y', 'u', method='n4sid', settings={'nk': 5})
```
