"""Momentum indicators: RSI, Stochastic, KDJ."""
from __future__ import annotations

import numpy as np

from .trend import sma


def rsi(series, window: int = 14) -> np.ndarray:
    """Wilder's RSI. Returns 100 for a flat/zero-loss series."""
    arr = np.asarray(series, dtype=float)
    n = arr.size
    out = np.full(n, np.nan)
    if n < window + 1:
        return out
    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(gains[:window].mean())
    avg_loss = float(losses[:window].mean())
    out[window] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(window + 1, n):
        avg_gain = (avg_gain * (window - 1) + gains[i - 1]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i - 1]) / window
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def stochastic(high, low, close, k_window: int = 14, d_window: int = 3):
    """Return ``(%K, %D)``. %D is the SMA of %K."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = close.size
    k = np.full(n, np.nan)
    for i in range(k_window - 1, n):
        hh = np.max(high[i - k_window + 1:i + 1])
        ll = np.min(low[i - k_window + 1:i + 1])
        k[i] = 50.0 if hh == ll else 100 * (close[i] - ll) / (hh - ll)
    d = sma(k, d_window)
    return k, d


def kdj(high, low, close, n: int = 9, k_smooth: float = 3.0, d_smooth: float = 3.0):
    """Return ``(K, D, J)`` using standard 2/3 smoothing (seeded at 50)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    size = close.size
    rsv = np.full(size, np.nan)
    for i in range(n - 1, size):
        hh = np.max(high[i - n + 1:i + 1])
        ll = np.min(low[i - n + 1:i + 1])
        rsv[i] = 50.0 if hh == ll else 100 * (close[i] - ll) / (hh - ll)
    K = np.full(size, np.nan)
    D = np.full(size, np.nan)
    J = np.full(size, np.nan)
    k_prev, d_prev = 50.0, 50.0
    for i in range(size):
        if np.isnan(rsv[i]):
            continue
        k_prev = (k_smooth - 1) / k_smooth * k_prev + 1 / k_smooth * rsv[i]
        d_prev = (d_smooth - 1) / d_smooth * d_prev + 1 / d_smooth * k_prev
        K[i], D[i] = k_prev, d_prev
        J[i] = 3 * k_prev - 2 * d_prev
    return K, D, J
