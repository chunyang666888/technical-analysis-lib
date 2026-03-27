"""Volume indicator: On-Balance Volume (OBV)."""
from __future__ import annotations

import numpy as np


def on_balance_volume(close, volume) -> np.ndarray:
    """Cumulative volume signed by the direction of close changes."""
    close = np.asarray(close, dtype=float)
    vol = np.asarray(volume, dtype=float)
    n = close.size
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - vol[i]
        else:
            obv[i] = obv[i - 1]
    return obv
