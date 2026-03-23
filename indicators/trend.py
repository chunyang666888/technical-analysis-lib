"""Trend indicators: SMA, EMA, MACD."""
from __future__ import annotations

import numpy as np


def sma(series, window: int) -> np.ndarray:
    """Simple moving average. Leading entries are NaN until ``window`` samples exist."""
    arr = np.asarray(series, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(arr.shape, np.nan)
    if arr.size < window:
        return out
    out[window - 1:] = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def ema(series, span: int) -> np.ndarray:
    """Exponential moving average (seeded with the first value)."""
    arr = np.asarray(series, dtype=float)
    n = arr.size
    out = np.full(n, np.nan)
    if n == 0 or span <= 0:
        return out
    k = 2.0 / (span + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def macd(series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return ``(macd_line, signal_line, histogram)`` as numpy arrays."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist
