# llsi
Lightweight Linear System Identification package.

llsi offers easy acess to system identification algorithms. Currently implemented are *n4sid*, *PO-MOESP* for state space identification, and *arx* for the identification of transfer function models. Additionally, a prediction error method (*pem*) exists for the identification of output-error (*oe*) models or iterative improvement of state-space models. llsi only depeds on numpy, scipy and matplotlib.

To try them out online, you can use [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminwitte/llsi/HEAD?labpath=notebooks%2Fexample.ipynb).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Usage
### Identification
1. Load data
start with loading the heated wire dataset (found in the data/ folder at the root of this repo) using numpy
```python
import numpy as np
d = np.load('heated_wire_data.npy')
```
2. Create a SysIdData object
```python
import llsi
data = llsi.SysIdData(t=d[:,0],Re=d[:,1],Nu=d[:,2])
```
the three data series are time (t), Reynolds number (Re) and Nußelt number (Nu). We are going to model the dynamics of the Nußelt number (heat transfer from wire to surrounding fluid) using Reynolds number (velocity of the surrounding fluid) as input.
3. Ensure the time steps are equidistant and the sampling rate is reasonable. Moreover, the beginning of the time series (transient start) is removed and finally the series are centerd around their respective mean value (which is a requirement for linear systems).
```python
data.equidistant()
data.downsample(3)
data.crop(start=100)
data.center()
```
4. Identify a state space model with order 3 using the "PO-MOESP" algorithm.
```python
mod = llsi.sysid(data,'Nu','Re',(3,),method='po-moesp')
```
5. Use it further with scipy by exporting it to a scipy.signal.StateSpace object
```python
ss = mod.to_ss()
```
or to a continuous time transfer function
```python
ss = mod.to_tf(continuous=True)
```

### Plotting
Optionally, if matplotlib is installed, simple plots can be created using the llsi.Figure context manager:
```python
with llsi.Figure() as fig:
    fig.plot(ss,'impulse')
```
will plot the impulse response of the model ss.

## Contribution
Thank you for considering to contribute. Any exchange and help is welcome. However, I have to ask you to be patient with me responding.
