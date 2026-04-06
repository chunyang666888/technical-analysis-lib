import numpy as np
from indicators import vwap, williams_r


def test_vwap_monotone_volume():
    high = [10, 11, 12, 13, 14.0]
    low = [9, 10, 11, 12, 13.0]
    close = [9.5, 10.5, 11.5, 12.5, 13.5]
    vol = [100, 100, 100, 100, 100.0]
    out = vwap(high, low, close, vol)
    # With equal volume, VWAP equals the typical-price average cumulatively.
    typ = (np.array(high) + np.array(low) + np.array(close)) / 3
    expected = np.cumsum(typ * 100) / np.cumsum(100 * np.ones(5))
    assert np.allclose(out, expected)


def test_vwap_first_nan_safe():
    out = vwap([1, 2], [0, 1], [1, 1.5], [0, 10])
    assert np.isnan(out[0])  # zero cumulative volume -> NaN
    assert not np.isnan(out[1])


def test_williams_r_bounds():
    high = [10, 11, 12, 13, 14, 15.0]
    low = [9, 10, 11, 12, 13, 14.0]
    close = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    w = williams_r(high, low, close, window=3)
    assert np.isnan(w[1])  # not enough window
    assert np.all(w[2:] >= -100.0)
    assert np.all(w[2:] <= 0.0)


def test_williams_r_closes_at_high_is_zero():
    # Price always at the high of its window -> %R = 0
    high = [10, 11, 12, 13.0]
    low = [9, 9, 9, 9.0]
    close = [10, 11, 12, 13.0]
    w = williams_r(high, low, close, window=3)
    assert abs(w[3] - 0.0) < 1e-9
