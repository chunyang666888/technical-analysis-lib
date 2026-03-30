"""Momentum indicator: Williams %R."""
from __future__ import annotations

import numpy as np


def williams_r(high, low, close, window: int = 14) -> np.ndarray:
    """Williams %R in the range [-100, 0].

    -100 means price closed at the period low (max bearish); 0 means at the
    period high (max bullish). Returns NaN until a full window is available.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = close.size
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        hh = np.max(high[i - window + 1:i + 1])
        ll = np.min(low[i - window + 1:i + 1])
        out[i] = -100.0 if hh == ll else -100 * (hh - close[i]) / (hh - ll)
    return out
