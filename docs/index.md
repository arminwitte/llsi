# llsi Documentation

Welcome to the documentation for **llsi** (Lightweight Linear System Identification).

## Overview

llsi offers easy access to system identification algorithms. Currently implemented are:
- **n4sid**: Subspace identification
- **PO-MOESP**: Subspace identification
- **arx**: AutoRegressive with eXogenous input
- **pem**: Prediction Error Method

## Installation

```bash
pip install llsi
```

## Usage

See the [README](../README.md) and the [notebooks](../notebooks/) folder for examples.

## API Reference

### Core Models
- `llsi.ltimodel`
- `llsi.statespacemodel`
- `llsi.polynomialmodel`

### Algorithms
- `llsi.sysid`
- `llsi.arx`
- `llsi.subspace`
- `llsi.pem`

### Utilities
- `llsi.SysIdData`
- `llsi.Figure`
