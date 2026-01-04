import pandas as pd
import numpy as np
from llsi.sysiddata import SysIdData


def make_irregular_times(start, periods, freq_seconds, jitter_seconds):
    # create roughly equidistant times with small jitter
    times = [start]
    for i in range(1, periods):
        times.append(times[-1] + np.timedelta64(int(freq_seconds + np.random.randint(-jitter_seconds, jitter_seconds + 1)), 's'))
    return np.array(times, dtype='datetime64[s]')


def test_from_logfile_basic(tmp_path):
    # Create a small logfile with two sensors and slightly irregular timestamps
    start = np.datetime64('2020-01-01T00:00:00')
    times = make_irregular_times(start, periods=6, freq_seconds=3600, jitter_seconds=60)

    data = []
    for t in times:
        data.append({'datetime': str(t), 'property_name': 'heating', 'temperature': float(np.random.rand() * 10)})
        data.append({'datetime': str(t), 'property_name': 'cooling', 'temperature': float(np.random.rand() * 5)})

    df = pd.DataFrame(data)
    p = tmp_path / "log.csv"
    df.to_csv(p, index=False)

    # Call from_logfile with 1 hour resample rule
    sid = SysIdData.from_logfile(str(p), resample_rule='1H')

    # Should be equidistant and contain both series
    assert sid.t is None
    assert 'heating' in sid.series
    assert 'cooling' in sid.series
    assert sid.N >= 2


def test_from_logfile_with_N(tmp_path):
    # Create deterministic data for testing N resizing
    start = np.datetime64('2020-01-01T00:00:00')
    times = np.array([start + np.timedelta64(i * 3600, 's') for i in range(4)], dtype='datetime64[s]')

    data = []
    for t in times:
        data.append({'datetime': str(t), 'property_name': 's', 'temperature': float(t.astype('datetime64[s]').astype(int) % 100)})

    df = pd.DataFrame(data)
    p = tmp_path / "log2.csv"
    df.to_csv(p, index=False)

    sid = SysIdData.from_logfile(str(p), resample_rule='1H', N=8)
    # Expect equidistant with 8 samples
    assert sid.t is None
    assert sid.N == 8


def test_from_logfile_custom_columns(tmp_path):
    # Test custom column names for time/value/pivot
    start = np.datetime64('2020-01-01T00:00:00')
    times = np.array([start + np.timedelta64(i * 3600, 's') for i in range(3)], dtype='datetime64[s]')

    data = []
    for t in times:
        data.append({'ts': str(t), 'sensor': 'A', 'val': 1.0})
        data.append({'ts': str(t), 'sensor': 'B', 'val': 2.0})

    df = pd.DataFrame(data)
    p = tmp_path / "log3.csv"
    df.to_csv(p, index=False)

    sid = SysIdData.from_logfile(str(p), resample_rule='1H', time_col='ts', value_col='val', pivot_col='sensor')
    assert 'A' in sid.series and 'B' in sid.series
    assert sid.N >= 3
