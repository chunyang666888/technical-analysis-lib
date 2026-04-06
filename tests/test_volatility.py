import numpy as np

from indicators.volatility import bollinger_bands, atr
from indicators.volume import on_balance_volume


def test_bollinger_constant_series():
    upper, mid, lower = bollinger_bands([5] * 20, window=10, num_std=2.0)
    assert abs(mid[-1] - 5.0) < 1e-9
    assert abs(upper[-1] - 5.0) < 1e-9
    assert abs(lower[-1] - 5.0) < 1e-9


def test_atr_flat_range_is_zero():
    flat = [10] * 20
    out = atr(flat, flat, flat, window=14)
    assert np.allclose(out[~np.isnan(out)], 0.0, atol=1e-9)


def test_obv_sign_follows_close():
    close = [10, 11, 12, 11, 12]
    volume = [1, 1, 1, 1, 1]
    obv = on_balance_volume(close, volume)
    assert list(obv) == [0, 1, 2, 1, 2]
