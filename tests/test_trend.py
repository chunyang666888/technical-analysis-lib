import numpy as np

from indicators.trend import sma, ema, macd


def test_sma_basic():
    out = sma([1, 2, 3, 4, 5], 2)
    assert np.isnan(out[0])
    assert np.allclose(out[1:], [1.5, 2.5, 3.5, 4.5])


def test_sma_invalid_window():
    import pytest
    with pytest.raises(ValueError):
        sma([1, 2, 3], 0)


def test_ema_constant_series():
    out = ema([5, 5, 5, 5], 3)
    assert np.allclose(out, 5.0)


def test_macd_constant_series_is_zero():
    line, signal, hist = macd([5, 5, 5, 5, 5, 5, 5, 5], 3, 6, 3)
    assert np.allclose(line, 0.0, atol=1e-9)
    assert np.allclose(signal, 0.0, atol=1e-9)
    assert np.allclose(hist, 0.0, atol=1e-9)
