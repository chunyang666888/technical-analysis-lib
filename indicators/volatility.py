"""Volatility indicators: Bollinger Bands, ATR."""
from __future__ import annotations

import numpy as np

from .trend import sma


def bollinger_bands(series, window: int = 20, num_std: float = 2.0):
    """Return ``(upper, middle, lower)`` bands."""
    arr = np.asarray(series, dtype=float)
    mid = sma(arr, window)
    std = np.full(arr.size, np.nan)
    for i in range(window - 1, arr.size):
        std[i] = float(np.std(arr[i - window + 1:i + 1], ddof=0))
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(high, low, close, window: int = 14) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = close.size
    out = np.full(n, np.nan)
    if n < 2:
        return out
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    if n < window + 1:
        return out
    out[window] = float(np.mean(tr[1:window + 1]))
    for i in range(window + 1, n):
        out[i] = (out[i - 1] * (window - 1) + tr[i]) / window
    return out
