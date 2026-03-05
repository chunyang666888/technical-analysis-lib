"""technical-analysis-lib — a typed, numpy-backed technical indicator toolkit.

Pure functions over array-likes: trend, momentum, volatility and volume
indicators, all returning ``numpy`` arrays with ``NaN`` where undefined.
No pandas required.
"""

from .trend import sma, ema, macd
from .momentum import rsi, stochastic, kdj
from .volatility import bollinger_bands, atr
from .volume import on_balance_volume
from .vwap import vwap
from .williams_r import williams_r

__all__ = [
    "sma", "ema", "macd",
    "rsi", "stochastic", "kdj",
    "bollinger_bands", "atr",
    "on_balance_volume",
    "vwap", "williams_r",
]
