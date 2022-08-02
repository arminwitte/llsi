# llsi
Lightweight Linear System Identification package.

llsi offers easy acess to system identification algorithms. Currently implemented are "n4sid", "PO-MOESP", and a prediction error method for state space identification ("PEM_SS") It only depeds on numpy, scipy and optionally matplotlib.

## Usage
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

## Contribution
Thank you for considering to contribute. Any exchange and help is welcome. However, I have to ask you to be patient with me responding.
