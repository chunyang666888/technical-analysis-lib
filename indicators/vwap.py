"""Volume indicator: Volume-Weighted Average Price (VWAP)."""
from __future__ import annotations

import numpy as np


def vwap(high, low, close, volume, anchor: str = "cumulative") -> np.ndarray:
    """Volume-Weighted Average Price.

    ``anchor="cumulative"`` returns the running VWAP over the whole series
    (typical session VWAP). ``anchor="daily"`` is not supported without a date
    index — pass pre-split intraday segments for per-session VWAP.

    Returns an array aligned with the inputs; the first value is NaN until a
    volume observation exists.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    vol = np.asarray(volume, dtype=float)
    n = close.size
    typical = (high + low + close) / 3.0
    pv = typical * vol
    cum_pv = np.cumsum(pv)
    cum_v = np.cumsum(vol)
    out = np.full(n, np.nan)
    mask = cum_v > 0
    out[mask] = cum_pv[mask] / cum_v[mask]
    return out
