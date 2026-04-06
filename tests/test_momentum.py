import numpy as np

from indicators.momentum import rsi, stochastic, kdj


def test_rsi_strong_uptrend_is_100():
    out = rsi(list(range(1, 21)), window=14)
    assert np.isnan(out[13])
    assert np.allclose(out[14:], 100.0)


def test_stochastic_at_high_is_100():
    close = list(range(1, 21))
    high = list(close)
    low = [c - 1 for c in close]
    k, d = stochastic(high, low, close, k_window=5, d_window=3)
    assert abs(k[-1] - 100.0) < 1e-6
    assert not np.isnan(d[-1])


def test_kdj_at_high_approaches_100():
    close = list(range(1, 21))
    high = list(close)
    low = [c - 1 for c in close]
    K, D, J = kdj(high, low, close, n=9)
    # With close pinned at the top of its range, RSV=100; the 2/3 EMA only
    # *approaches* 100 (D lags K), so assert all lines are strongly bullish.
    assert K[-1] > 99.0
    assert D[-1] > 95.0
    assert J[-1] > 99.0
